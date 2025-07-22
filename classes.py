class DynaLRPlusPlusAdaptivePID: 
    def __init__(self, optimizer, I=80, I_min=40, I_max=400,
                 kP_base=0.25, kI_base=0.15, kD_base=0.08,
                 rho=0.18, min_factor=0.7, max_factor=1.3,
                 clip_lr=(5e-4, 5e-2), warmup_steps=500,
                 momentum_threshold=0.05):
        self.opt = optimizer
        self.I, self.I_min, self.I_max = I, I_min, I_max
        self.kP_base, self.kI_base, self.kD_base = kP_base, kI_base, kD_base
        self.rho = rho
        self.min_factor, self.max_factor = min_factor, max_factor
        self.lr_min, self.lr_max = clip_lr
        self.warmup_steps = warmup_steps
        self.momentum_threshold = momentum_threshold

        self.step_count = 0
        self.adjust_count = 0
        self.best_loss = float('inf')
        ng = len(optimizer.param_groups)
        self.prev_ema = [None] * ng
        self.integral = [0.0] * ng
        self.prev_delta = [0.0] * ng
        self.loss_momentum = [0.0] * ng
        self.last_ema = [None] * ng

        self._update_gains()

    def _update_gains(self):
        progress = min(1.0, self.step_count / 20000)
        self.kP = self.kP_base * (1.4 - 0.4 * progress)
        self.kI = self.kI_base * (0.8 + 0.6 * progress)
        self.kD = self.kD_base * (1.3 - 0.5 * progress)

    def _update_momentum(self, curr_loss, idx):
        if self.prev_ema[idx] is None:
            self.loss_momentum[idx] = 0
            return
        change = curr_loss - self.prev_ema[idx]
        factor = 1.5 if abs(change) > self.momentum_threshold else 0.8
        self.loss_momentum[idx] = factor * self.loss_momentum[idx] + (1 - factor) * change

    def update_loss(self, loss):
        self.step_count += 1
        curr = loss.item()
        self._update_gains()

        for i in range(len(self.prev_ema)):
            if self.prev_ema[i] is None:
                self.prev_ema[i] = curr
                self.last_ema[i] = curr
            else:
                smoothing = 0.8 if abs(curr - self.prev_ema[i]) > 0.1 * self.prev_ema[i] else 0.95
                self.prev_ema[i] = smoothing * self.prev_ema[i] + (1 - smoothing) * curr
            self._update_momentum(curr, i)

        if self.step_count < self.warmup_steps or self.step_count % self.I != 0:
            return

        rewards = []
        for i, pg in enumerate(self.opt.param_groups):
            delta = self.prev_ema[i] - self.last_ema[i]
            self.last_ema[i] = self.prev_ema[i]
            P = self.kP * (delta + 0.3 * self.loss_momentum[i])
            D = self.kD * (delta - self.prev_delta[i])
            if delta < 0:
                self.integral[i] = 0.98 * self.integral[i] + delta
            else:
                self.integral[i] *= 0.7
            I = self.kI * self.integral[i]
            reward = P + I - 0.7 * D
            rewards.append(reward)
            self.prev_delta[i] = delta

        mean_r = sum(rewards) / len(rewards)
        if mean_r > self.rho:
            self.I = max(self.I_min, int(self.I * 0.85))
        else:
            self.I = min(self.I_max, int(self.I * 1.15))

        for i, pg in enumerate(self.opt.param_groups):
            momentum_boost = 1.0 + 0.4 * abs(self.loss_momentum[i])
            factor = momentum_boost * (1 + self.kP * rewards[i])
            factor = max(self.min_factor, min(self.max_factor, factor))
            new_lr = pg['lr'] * factor
            pg['lr'] = max(self.lr_min, min(self.lr_max, new_lr))

        current_loss = sum(self.prev_ema) / len(self.prev_ema)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        self.adjust_count += 1

    def step(self, loss=None):
        if loss is not None:
            self.update_loss(loss)
        self.opt.step()


class DynaLRPlusPlusNoMemory:
    def __init__(self, optimizer, I=100, I_min=50, I_max=500,
                 gamma_grow=1.2, gamma_shrink=1.2,
                 kP=0.1, kI=0.01, kD=0.05, rho=0.15):
        self.opt = optimizer
        self.I, self.I_min, self.I_max = I, I_min, I_max
        self.gamma_grow, self.gamma_shrink = gamma_grow, gamma_shrink
        self.kP, self.kI, self.kD, self.rho = kP, kI, kD, rho

        ng = len(optimizer.param_groups)
        self.t = 0
        self.prev_emas = [None] * ng
        self.integrals = [0.0] * ng
        self.prev_deltas = [0.0] * ng

    def update_loss(self, loss):
        self.t += 1
        current = loss.item()
        old = self.prev_emas.copy()
        for i in range(len(self.prev_emas)):
            prev = old[i] if old[i] is not None else current
            self.prev_emas[i] = self.kP * current + (1 - self.kP) * prev
        if self.t % self.I != 0:
            return

        deltas = [old[i] - self.prev_emas[i] for i in range(len(old))]
        rewards = []
        for i, delta in enumerate(deltas):
            P = delta
            D = self.kD * (delta - self.prev_deltas[i])
            self.integrals[i] += delta
            I = self.kI * self.integrals[i]
            reward = P + I - D
            rewards.append(reward)
            self.prev_deltas[i] = delta

        max_r = max(rewards)
        for i, pg in enumerate(self.opt.param_groups):
            r = rewards[i]
            if abs(r - max_r) < self.rho * abs(max_r):
                factor = 1.05 if r > 0 else 0.95
            else:
                factor = 1.1 if r > (1 - self.rho) * max_r else 0.9
            pg['lr'] = max(1e-6, min(pg['lr'] * factor, 1e-1))

        mean_r = sum(rewards) / len(rewards)
        if mean_r > self.rho:
            self.I = max(self.I_min, int(self.I / self.gamma_shrink))
        else:
            self.I = min(self.I_max, int(self.I * self.gamma_grow))

    def step(self, loss=None):
        if loss is not None:
            self.update_loss(loss)
        self.opt.step()


class DynaLRPlusPlusMemory:
    def __init__(self, optimizer, I=100, I_min=50, I_max=500,
                 gamma_grow=1.2, gamma_shrink=1.2,
                 kP=0.1, kI=0.01, kD=0.05, rho=0.15):
        self.opt = optimizer
        self.I, self.I_min, self.I_max = I, I_min, I_max
        self.gamma_grow, self.gamma_shrink = gamma_grow, gamma_shrink
        self.kP, self.kI, self.kD, self.rho = kP, kI, kD, rho

        ng = len(optimizer.param_groups)
        self.t = 0
        self.prev_emas = [None] * ng
        self.integrals = [0.0] * ng
        self.prev_deltas = [0.0] * ng
        self.lr_memory = [{} for _ in range(ng)]

    def update_loss(self, loss):
        self.t += 1
        current = loss.item()
        old = self.prev_emas.copy()
        for i in range(len(self.prev_emas)):
            prev = old[i] if old[i] is not None else current
            self.prev_emas[i] = self.kP * current + (1 - self.kP) * prev
        if self.t % self.I != 0:
            return

        deltas = [old[i] - self.prev_emas[i] for i in range(len(old))]
        rewards = []
        for i, delta in enumerate(deltas):
            P = delta
            D = self.kD * (delta - self.prev_deltas[i])
            self.integrals[i] += delta
            I = self.kI * self.integrals[i]
            reward = P + I - D
            rewards.append(reward)
            self.prev_deltas[i] = delta

        max_r = max(rewards)
        for i, pg in enumerate(self.opt.param_groups):
            lr = pg['lr']
            r = rewards[i]
            mem = self.lr_memory[i]
            key = round(lr, 6)
            mem[key] = 0.9 * mem.get(key, r) + 0.1 * r
            best_known = max(mem.values())
            if abs(r - best_known) < self.rho * abs(best_known):
                factor = 1.05 if r > 0 else 0.95
            else:
                factor = 1.1 if r > (1 - self.rho) * max_r else 0.9
            pg['lr'] = max(1e-6, min(lr * factor, 1e-1))

        mean_r = sum(rewards) / len(rewards)
        if mean_r > self.rho:
            self.I = max(self.I_min, int(self.I / self.gamma_shrink))
        else:
            self.I = min(self.I_max, int(self.I * self.gamma_grow))

    def step(self, loss=None):
        if loss is not None:
            self.update_loss(loss)
        self.opt.step()


class DynaLRPlusPlusEnhanced:
    def __init__(self, optimizer, I=100, I_min=50, I_max=500,
                 gamma_grow=1.2, gamma_shrink=1.2,
                 kP=0.1, kI=0.01, kD=0.05,
                 rho=0.15, epsilon=0.05):
        self.opt = optimizer
        self.I, self.I_min, self.I_max = I, I_min, I_max
        self.gamma_grow, self.gamma_shrink = gamma_grow, gamma_shrink
        self.kP, self.kI, self.kD, self.rho = kP, kI, kD, rho
        self.epsilon = epsilon

        ng = len(optimizer.param_groups)
        self.t = 0
        self.prev_emas = [None] * ng
        self.integrals = [0.0] * ng
        self.prev_deltas = [0.0] * ng
        self.prev_grad_norms = [None] * ng

    def update_loss(self, loss):
        self.t += 1
        current = loss.item()
        old = self.prev_emas.copy()
        for i in range(len(self.prev_emas)):
            prev = old[i] if old[i] is not None else current
            self.prev_emas[i] = self.kP * current + (1 - self.kP) * prev
        if self.t % self.I != 0:
            return

        deltas = [old[i] - self.prev_emas[i] for i in range(len(old))]
        rewards = []
        for i, delta in enumerate(deltas):
            P = delta
            D = self.kD * (delta - self.prev_deltas[i])
            self.integrals[i] += delta
            I = self.kI * self.integrals[i]
            reward = P + I - D
            rewards.append(reward)
            self.prev_deltas[i] = delta

        max_r = max(rewards)
        for i, pg in enumerate(self.opt.param_groups):
            norms = [p.grad.data.norm() for p in pg['params'] if p.grad is not None]
            gn = torch.stack(norms).mean().item() if norms else 0.0
            prev_gn = self.prev_grad_norms[i] if self.prev_grad_norms[i] is not None else gn
            grad_reward = prev_gn - gn
            self.prev_grad_norms[i] = gn
            r = rewards[i] + 0.5 * grad_reward

            if random.random() < self.epsilon:
                factor = 1 + (random.random() * 0.2 - 0.1)
            else:
                factor = 1.05 if abs(r - max_r) < self.rho * abs(max_r) and r > 0 else 0.9 if abs(r - max_r) < self.rho * abs(max_r) else 1.1 if r > (1 - self.rho) * max_r else 0.9
            pg['lr'] = max(1e-6, min(pg['lr'] * factor, 1e-1))

        mean_r = sum(rewards) / len(rewards)
        if mean_r > self.rho:
            self.I = max(self.I_min, int(self.I / self.gamma_shrink))
        else:
            self.I = min(self.I_max, int(self.I * self.gamma_grow))

    def step(self, loss=None):
        if loss is not None:
            self.update_loss(loss)
        self.opt.step()
