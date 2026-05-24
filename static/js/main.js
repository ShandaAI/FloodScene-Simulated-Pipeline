class MotionApp {
    constructor() {
        this.sessionId = null;
        this.realtimeSocket = null;

        this.isRunning = false;
        this.isPaused = false;
        this.inputMode = 'online';
        this.targetFps = 20;
        this.frameCount = 0;
        this.motionFpsCounter = 0;
        this.motionFpsUpdatedAt = performance.now();
        this.timerStartedAt = null;
        this.timerStoppedAt = null;
        this.bufferCapacity = 4;
        this.historyLength = 4;
        this.smoothingAlpha = 1.0;
        this.generationSeed = 11;
        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000;
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        this.displayFrameStartedAt = 0;
        this.sourceFrameIntervalMs = 1000 / 30;
        this.avatar = null;
        this.meshReadyPromise = Promise.resolve();
        this.config = null;
        this.offlineCueRows = [
            { text: '江南style 跳舞', start: 0, end: '' },
            { text: '鞠躬', start: 8, end: 12 },
        ];

        this.bindElements();
        this.initThreeJS();
        this.initUI();
        this.meshReadyPromise = this.loadConfig();
        this.animate();
    }

    bindElements() {
        this.statusEl = document.getElementById('status');
        this.bufferSizeEl = document.getElementById('bufferSize');
        this.fpsEl = document.getElementById('fps');
        this.frameCountEl = document.getElementById('frameCount');
        this.elapsedTimeEl = document.getElementById('elapsedTime');
        this.currentSmoothingEl = document.getElementById('currentSmoothing');
        this.currentHistoryEl = document.getElementById('currentHistory');
        this.motionText = document.getElementById('motionText');
        this.onlineModeBtn = document.getElementById('onlineModeBtn');
        this.offlineModeBtn = document.getElementById('offlineModeBtn');
        this.onlineInputPanel = document.getElementById('onlineInputPanel');
        this.offlineInputPanel = document.getElementById('offlineInputPanel');
        this.offlineScheduleRows = document.getElementById('offlineScheduleRows');
        this.addCueBtn = document.getElementById('addCueBtn');
        this.startResetBtn = document.getElementById('startResetBtn');
        this.updateBtn = document.getElementById('updateBtn');
        this.pauseResumeBtn = document.getElementById('pauseResumeBtn');
        this.configBtn = document.getElementById('configBtn');
        this.configModal = document.getElementById('configModal');
        this.scheduleConfigFields = document.getElementById('scheduleConfigFields');
        this.cfgConfigFields = document.getElementById('cfgConfigFields');
        this.modalHistoryLength = document.getElementById('modalHistoryLength');
        this.modalSeed = document.getElementById('modalSeed');
        this.modalSmoothingAlpha = document.getElementById('modalSmoothingAlpha');
        this.modalSmoothingValue = document.getElementById('modalSmoothingValue');
        this.configDiscardBtn = document.getElementById('configDiscardBtn');
        this.configSaveBtn = document.getElementById('configSaveBtn');
    }

    initThreeJS() {
        const canvas = document.getElementById('renderCanvas');
        const container = document.getElementById('canvas-container');

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        this.camera = new THREE.PerspectiveCamera(
            48,
            container.clientWidth / container.clientHeight,
            0.1,
            1000,
        );
        this.camera.position.set(4.8, 2.3, 5.0);
        this.camera.lookAt(0, 0.9, 0);

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));

        this.keyLight = new THREE.DirectionalLight(0xffffff, 0.82);
        this.keyLight.position.set(5, 8, 3);
        this.keyLight.castShadow = true;
        this.keyLight.shadow.mapSize.set(2048, 2048);
        this.keyLight.shadow.camera.near = 0.5;
        this.keyLight.shadow.camera.far = 50;
        this.keyLight.shadow.camera.left = -5;
        this.keyLight.shadow.camera.right = 5;
        this.keyLight.shadow.camera.top = 5;
        this.keyLight.shadow.camera.bottom = -5;
        this.scene.add(this.keyLight);
        this.scene.add(this.keyLight.target);

        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(22, 22),
            new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.08 }),
        );
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);

        const grid = new THREE.GridHelper(22, 22, 0xd0d0d0, 0xeeeeee);
        grid.position.y = 0.01;
        this.scene.add(grid);

        const targetRing = new THREE.Mesh(
            new THREE.RingGeometry(1.6, 1.62, 96),
            new THREE.MeshBasicMaterial({ color: 0xd6d6d6, transparent: true, opacity: 0.9 }),
        );
        targetRing.rotation.x = -Math.PI / 2;
        targetRing.position.y = 0.018;
        this.scene.add(targetRing);

        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 0.9, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;

        const markInteraction = () => {
            this.lastUserInteraction = performance.now();
        };
        canvas.addEventListener('pointerdown', markInteraction);
        canvas.addEventListener('wheel', markInteraction, { passive: true });

        window.addEventListener('resize', () => this.resize());
    }

    initUI() {
        this.startResetBtn.addEventListener('click', () => {
            if (this.isRunning) {
                this.reset();
            } else {
                this.start();
            }
        });
        this.updateBtn.addEventListener('click', () => this.updateText());
        this.pauseResumeBtn.addEventListener('click', () => this.togglePause());
        this.configBtn.addEventListener('click', () => this.openConfig());
        this.onlineModeBtn.addEventListener('click', () => this.setInputMode('online'));
        this.offlineModeBtn.addEventListener('click', () => this.setInputMode('offline'));
        this.addCueBtn.addEventListener('click', () => this.addOfflineCue());
        this.motionText.addEventListener('keydown', event => {
            if (event.key === 'Enter' && this.inputMode === 'online') {
                event.preventDefault();
                if (this.isRunning) this.updateText();
            }
        });
        this.configDiscardBtn.addEventListener('click', () => this.closeConfig());
        this.configSaveBtn.addEventListener('click', () => this.saveConfig());
        this.modalSmoothingAlpha.addEventListener('input', () => {
            this.modalSmoothingValue.textContent = Number(this.modalSmoothingAlpha.value).toFixed(2);
        });
        this.configModal.addEventListener('click', event => {
            if (event.target === this.configModal) this.closeConfig();
        });
        window.addEventListener('beforeunload', () => this.sendResetBeacon());
        this.renderStaticConfigFields();
        this.renderOfflineSchedule();
        this.setInputMode('online');
        this.syncConfigLabels();
    }

    renderStaticConfigFields() {
        this.scheduleConfigFields.innerHTML = `
            <div class="config-field">
                <label for="modalTargetFps">Target FPS</label>
                <input id="modalTargetFps" type="number" value="20" disabled>
            </div>
            <div class="config-field">
                <label for="modalBufferCapacity">Buffer Capacity</label>
                <input id="modalBufferCapacity" type="number" value="4" disabled>
            </div>
        `;
        this.cfgConfigFields.innerHTML = `
            <div class="config-field">
                <label for="modalGuidanceScale">Guidance Scale</label>
                <input id="modalGuidanceScale" type="text" value="simulated endpoint" disabled>
            </div>
        `;
    }

    setInputMode(mode) {
        if (this.isRunning) return;
        this.inputMode = mode;
        const isOnline = mode === 'online';
        this.onlineModeBtn.classList.toggle('active', isOnline);
        this.offlineModeBtn.classList.toggle('active', !isOnline);
        this.onlineInputPanel.hidden = !isOnline;
        this.offlineInputPanel.hidden = isOnline;
        this.updateBtn.textContent = isOnline ? 'Send Text' : 'Schedule Locked';
        this.updateBtn.disabled = !this.isRunning || !isOnline;
    }

    renderOfflineSchedule() {
        this.offlineScheduleRows.innerHTML = '';
        this.offlineCueRows.forEach((cue, index) => {
            const row = document.createElement('div');
            row.className = 'offline-cue-row';
            row.dataset.index = String(index);
            row.innerHTML = `
                <input class="offline-text" type="text" value="${this.escapeAttr(cue.text)}" placeholder="Text ${index + 1}">
                <input class="offline-start" type="number" min="0" step="0.1" value="${cue.start}" placeholder="Start">
                <input class="offline-end" type="number" min="0" step="0.1" value="${cue.end ?? ''}" placeholder="${index === this.offlineCueRows.length - 1 ? 'End' : 'Auto'}">
                <button class="btn btn-compact remove-cue" type="button"${this.offlineCueRows.length <= 1 ? ' disabled' : ''}>Remove</button>
            `;
            row.querySelector('.remove-cue').addEventListener('click', () => this.removeOfflineCue(index));
            this.offlineScheduleRows.appendChild(row);
        });
    }

    escapeAttr(value) {
        return String(value).replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;');
    }

    addOfflineCue() {
        const last = this.offlineCueRows[this.offlineCueRows.length - 1] || { start: 0 };
        this.offlineCueRows.push({
            text: 'new motion',
            start: Number(last.start || 0) + 4,
            end: Number(last.start || 0) + 8,
        });
        this.renderOfflineSchedule();
    }

    removeOfflineCue(index) {
        if (this.offlineCueRows.length <= 1) return;
        this.offlineCueRows.splice(index, 1);
        this.renderOfflineSchedule();
    }

    readOfflineSchedule() {
        const rows = [...this.offlineScheduleRows.querySelectorAll('.offline-cue-row')];
        const schedule = rows.map((row, index) => {
            const text = row.querySelector('.offline-text').value.trim();
            const start = Number(row.querySelector('.offline-start').value);
            const endInput = row.querySelector('.offline-end').value;
            const cue = { text, start };
            if (index === rows.length - 1) cue.end = Number(endInput);
            return cue;
        });

        if (!schedule.length) throw new Error('Offline schedule is empty.');
        if (schedule[0].start !== 0) throw new Error('The first offline cue must start at 0.');
        for (let i = 0; i < schedule.length; i++) {
            const cue = schedule[i];
            if (!cue.text) throw new Error('Offline cue text cannot be empty.');
            if (!Number.isFinite(cue.start) || cue.start < 0) throw new Error('Invalid offline cue start time.');
            if (i > 0 && cue.start <= schedule[i - 1].start) {
                throw new Error('Offline cue start times must increase.');
            }
        }
        const finalCue = schedule[schedule.length - 1];
        if (!Number.isFinite(finalCue.end) || finalCue.end <= finalCue.start) {
            throw new Error('The final offline cue must have an end time after its start.');
        }
        this.offlineCueRows = schedule.map(cue => ({ text: cue.text, start: cue.start, end: cue.end ?? '' }));
        return schedule;
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            this.config = config;
            this.targetFps = config.frame_rate || 20;
            if (Number.isInteger(config.kimodo_g1_default_seed)) {
                this.generationSeed = config.kimodo_g1_default_seed;
            }
            await this.initAvatar(config);
            this.syncConfigLabels();
            return config;
        } catch (error) {
            this.setStatus('Offline');
            throw error;
        }
    }

    async initAvatar(config) {
        if (this.avatar) return;
        this.avatar = AvatarFactory.fromConfig(config, this.scene);
        await this.avatar.loadTopology().catch(error => {
            this.setStatus(`${this.avatar.displayName} Missing`);
            throw error;
        });
    }

    wsUrl(path) {
        const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${scheme}//${window.location.host}${path}`;
    }

    async start() {
        this.setStatus('Loading');
        this.setControlsLocked(true);
        this.startTimer();

        try {
            await this.meshReadyPromise;
            if (!this.avatar) throw new Error('Avatar failed to initialize');
            this.frameCount = 0;
            this.motionFpsCounter = 0;
            this.motionFpsUpdatedAt = performance.now();
            this.displayFramePrevious = null;
            this.displayFrameCurrent = null;
            this.avatar.clearTrail();
            this.avatar.setVisible(false);
            this.updateFrameDisplay(0);
            this.updateBufferDisplay(0, this.bufferCapacity);
            this.fpsEl.textContent = '0';

            const sessionPayload = {
                renderer: this.config?.renderer || 'g1',
                input_mode: this.inputMode,
                frame_rate: this.targetFps,
                seed: this.generationSeed,
            };
            if (this.inputMode === 'online') {
                sessionPayload.initial_text = this.motionText.value;
            } else {
                sessionPayload.schedule = this.readOfflineSchedule();
            }

            const response = await fetch('/api/realtime/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sessionPayload),
            });

            if (!response.ok) {
                const error = await response.json();
                this.stopTimer();
                this.setStatus(error.detail || 'Error');
                this.setControlsLocked(false);
                return;
            }

            const data = await response.json();
            this.sessionId = data.session_id;
            this.isRunning = true;
            this.isPaused = false;
            this.connectRealtime();
            this.setButtonState();
        } catch (error) {
            this.stopTimer();
            this.setStatus(error.message || 'Error');
            this.setControlsLocked(false);
        }
    }

    connectRealtime() {
        const socket = new WebSocket(this.wsUrl(`/api/realtime/sessions/${this.sessionId}`));
        socket.binaryType = 'arraybuffer';
        this.realtimeSocket = socket;
        socket.onopen = () => {
            if (this.realtimeSocket !== socket) return;
            this.setStatus('Streaming');
        };
        socket.onmessage = event => {
            if (this.realtimeSocket !== socket) return;
            if (event.data instanceof ArrayBuffer) {
                this.applyBinaryMotionFrame(new Float32Array(event.data));
                return;
            }
            const data = JSON.parse(event.data);
            this.applyRealtimeEvent(data);
        };
        socket.onclose = () => {
            if (this.realtimeSocket !== socket) return;
            this.realtimeSocket = null;
            if (this.isRunning) {
                this.stopTimer();
                this.setStatus('Idle');
            }
        };
    }

    applyRealtimeEvent(data) {
        if (data.type === 'session.paused') {
            this.isPaused = true;
            this.setButtonState();
            this.setStatus('Paused');
        } else if (data.type === 'session.resumed' || data.type === 'session.started') {
            this.isPaused = false;
            this.setButtonState();
            this.setStatus('Streaming');
        } else if (data.type === 'input_text.committed') {
            this.setStatus(this.isPaused ? 'Paused' : 'Streaming');
        } else if (data.type === 'motion_generation.started') {
            this.setStatus('Generating');
        } else if (data.type === 'motion_generation.segment_completed') {
            this.setStatus('Streaming');
        } else if (data.type === 'motion_generation.completed') {
            this.setStatus('Streaming');
        } else if (data.type === 'offline_cue.changed') {
            this.setStatus('Streaming');
        } else if (data.type === 'offline_schedule.completed') {
            this.isRunning = false;
            this.isPaused = false;
            this.sessionId = null;
            this.stopTimer();
            this.closeSockets();
            this.setButtonState();
            this.setStatus('Complete');
        } else if (data.type === 'budget_exhausted') {
            this.stopTimer();
            this.setStatus('Budget');
            this.reset(false);
        } else if (data.type === 'error') {
            this.stopTimer();
            this.setStatus(data.code || 'Error');
        }
    }

    applyBinaryMotionFrame(packet) {
        const headerSize = 9;
        if (packet.length <= headerSize) return;
        if (!this.avatar) return;
        const frame = this.avatar.readFrame(packet, headerSize);
        if (!frame) return;

        this.enqueueDisplayFrame(frame);
        this.frameCount = frame.frameId;
        this.motionFpsCounter += 1;
        this.updateFrameDisplay(frame.frameId);
        this.updateBufferDisplay(frame.bufferSize, frame.bufferCapacity);
        this.updateMotionFps();
        if (!this.isPaused) this.setStatus('Streaming');
    }

    copyFrame(frame) {
        return {
            frameId: frame.frameId,
            root: [...frame.root],
            bufferSize: frame.bufferSize,
            bufferCapacity: frame.bufferCapacity,
            joints: new Float32Array(frame.joints),
            rotations: new Float32Array(frame.rotations),
        };
    }

    enqueueDisplayFrame(frame) {
        const copied = this.copyFrame(frame);
        if (!this.displayFrameCurrent) {
            this.displayFramePrevious = copied;
            this.displayFrameCurrent = copied;
            this.displayFrameStartedAt = performance.now();
            this.avatar.applyFrame(copied, 1);
            this.currentRootPos.set(copied.root[0], copied.root[1], copied.root[2]);
            return;
        }
        this.displayFramePrevious = this.displayFrameCurrent;
        this.displayFrameCurrent = copied;
        this.displayFrameStartedAt = performance.now();
    }

    applyInterpolatedDisplayFrame(now) {
        if (!this.avatar || !this.displayFrameCurrent) return;
        if (!this.displayFramePrevious) {
            this.avatar.applyFrame(this.displayFrameCurrent, 1);
            return;
        }

        const alpha = Math.max(0, Math.min(1, (now - this.displayFrameStartedAt) / this.sourceFrameIntervalMs));
        const prev = this.displayFramePrevious;
        const curr = this.displayFrameCurrent;
        const joints = new Float32Array(curr.joints.length);
        for (let i = 0; i < joints.length; i++) {
            joints[i] = prev.joints[i] * (1 - alpha) + curr.joints[i] * alpha;
        }
        const root = [
            prev.root[0] * (1 - alpha) + curr.root[0] * alpha,
            prev.root[1] * (1 - alpha) + curr.root[1] * alpha,
            prev.root[2] * (1 - alpha) + curr.root[2] * alpha,
        ];
        this.avatar.applyFrame({ ...curr, root, joints }, 1);
        this.currentRootPos.set(root[0], root[1], root[2]);
    }

    async updateText() {
        if (!this.sessionId || this.inputMode !== 'online') return;
        const text = this.motionText.value.trim();
        if (!text) return;
        try {
            if (this.realtimeSocket && this.realtimeSocket.readyState === WebSocket.OPEN) {
                this.realtimeSocket.send(JSON.stringify({ type: 'input_text.append', text }));
            } else {
                const response = await fetch(`/api/realtime/sessions/${this.sessionId}/input_text`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text }),
                });
                if (!response.ok) throw new Error('Text update failed');
            }
            this.setStatus(this.isPaused ? 'Paused' : 'Streaming');
        } catch (error) {
            this.setStatus('Error');
        }
    }

    async togglePause() {
        if (!this.sessionId) return;
        const nextPaused = !this.isPaused;
        try {
            if (this.realtimeSocket && this.realtimeSocket.readyState === WebSocket.OPEN) {
                this.realtimeSocket.send(JSON.stringify({ type: nextPaused ? 'session.pause' : 'session.resume' }));
            }
        } catch (error) {
            this.setStatus('Error');
        }
    }

    async reset(callApi = true) {
        const sessionId = this.sessionId;
        this.isRunning = false;
        this.isPaused = false;
        this.sessionId = null;
        this.closeSockets();
        this.clearTimer();
        if (callApi && sessionId) {
            try {
                await fetch(`/api/realtime/sessions/${sessionId}/close`, { method: 'POST' });
            } catch (error) {
                // Reset is best-effort because closed sockets already stop the local view.
            }
        }
        this.setStatus('Idle');
        this.frameCount = 0;
        this.motionFpsCounter = 0;
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        if (this.avatar) {
            this.avatar.clearTrail();
            this.avatar.setVisible(false);
        }
        this.updateFrameDisplay(0);
        this.fpsEl.textContent = '0';
        this.updateBufferDisplay(0, this.bufferCapacity);
        this.setButtonState();
    }

    sendResetBeacon() {
        if (!this.sessionId) return;
        navigator.sendBeacon(`/api/realtime/sessions/${this.sessionId}/close`, new Blob([], { type: 'application/json' }));
    }

    closeSockets() {
        if (this.realtimeSocket && this.realtimeSocket.readyState <= WebSocket.OPEN) {
            if (this.realtimeSocket.readyState === WebSocket.OPEN) {
                this.realtimeSocket.send(JSON.stringify({ type: 'session.close' }));
            }
            this.realtimeSocket.close();
        }
        this.realtimeSocket = null;
    }

    updateMotionFps() {
        const now = performance.now();
        if (now - this.motionFpsUpdatedAt > 1000) {
            this.fpsEl.textContent = String(this.motionFpsCounter);
            this.motionFpsCounter = 0;
            this.motionFpsUpdatedAt = now;
        }
    }

    updateFrameDisplay(frameId) {
        this.frameCountEl.textContent = String(frameId);
    }

    updateBufferDisplay(size, capacity) {
        const bufferSize = Number.isFinite(size) ? size : Math.min(this.bufferCapacity, this.frameCount);
        const bufferCapacity = Number.isFinite(capacity) ? capacity : this.bufferCapacity;
        this.bufferCapacity = bufferCapacity;
        this.bufferSizeEl.textContent = `${Math.min(bufferSize, bufferCapacity)} / ${bufferCapacity}`;
    }

    openConfig() {
        this.modalHistoryLength.value = String(this.historyLength);
        this.modalSeed.value = String(this.generationSeed);
        this.modalSmoothingAlpha.value = String(this.smoothingAlpha);
        this.modalSmoothingValue.textContent = this.smoothingAlpha.toFixed(2);
        this.configModal.hidden = false;
    }

    closeConfig() {
        this.configModal.hidden = true;
    }

    async saveConfig() {
        const nextHistory = Math.max(1, Math.min(16, Number(this.modalHistoryLength.value) || 4));
        const nextSeed = Math.max(0, Math.min(2147483647, Math.floor(Number(this.modalSeed.value) || 0)));
        const nextSmoothing = Math.max(0, Math.min(1, Number(this.modalSmoothingAlpha.value)));
        const shouldRestart = this.isRunning;
        this.historyLength = nextHistory;
        this.generationSeed = nextSeed;
        this.smoothingAlpha = nextSmoothing;
        this.syncConfigLabels();
        this.closeConfig();
        if (shouldRestart) {
            await this.reset();
            await this.start();
        }
    }

    syncConfigLabels() {
        this.currentSmoothingEl.textContent = this.smoothingAlpha.toFixed(2);
        this.currentHistoryEl.textContent = String(this.historyLength);
    }

    setStatus(status) {
        this.statusEl.textContent = status;
    }

    startTimer() {
        this.timerStartedAt = performance.now();
        this.timerStoppedAt = null;
        this.updateElapsedDisplay();
    }

    stopTimer() {
        if (this.timerStartedAt !== null && this.timerStoppedAt === null) {
            this.timerStoppedAt = performance.now();
            this.updateElapsedDisplay();
        }
    }

    clearTimer() {
        this.timerStartedAt = null;
        this.timerStoppedAt = null;
        this.updateElapsedDisplay();
    }

    updateElapsedDisplay(now = performance.now()) {
        if (!this.elapsedTimeEl) return;
        if (this.timerStartedAt === null) {
            this.elapsedTimeEl.textContent = '0.00s';
            return;
        }
        const end = this.timerStoppedAt ?? now;
        const seconds = Math.max(0, (end - this.timerStartedAt) / 1000);
        this.elapsedTimeEl.textContent = `${seconds.toFixed(2)}s`;
    }

    setControlsLocked(locked) {
        this.startResetBtn.disabled = locked;
        this.updateBtn.disabled = locked || !this.isRunning || this.inputMode !== 'online';
        this.pauseResumeBtn.disabled = locked || !this.isRunning;
        this.configBtn.disabled = false;
    }

    setButtonState() {
        this.startResetBtn.disabled = false;
        this.startResetBtn.textContent = this.isRunning ? 'Reset' : 'Start';
        this.updateBtn.disabled = !this.isRunning || this.inputMode !== 'online';
        this.pauseResumeBtn.disabled = !this.isRunning;
        this.pauseResumeBtn.textContent = this.isPaused ? 'Resume' : 'Pause';
        this.pauseResumeBtn.classList.toggle('btn-success', this.isPaused);
        this.pauseResumeBtn.classList.toggle('btn-warning', !this.isPaused);
        this.onlineModeBtn.disabled = this.isRunning;
        this.offlineModeBtn.disabled = this.isRunning;
        this.addCueBtn.disabled = this.isRunning;
        this.offlineScheduleRows.querySelectorAll('input, button').forEach(element => {
            element.disabled = this.isRunning;
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        const now = performance.now();
        if (this.timerStartedAt !== null && this.timerStoppedAt === null) {
            this.updateElapsedDisplay(now);
        }
        this.applyInterpolatedDisplayFrame(now);
        if (now - this.lastUserInteraction > this.autoFollowDelay) {
            this.controls.target.lerp(
                new THREE.Vector3(this.currentRootPos.x, 0.9, this.currentRootPos.z),
                0.035,
            );
        }
        this.keyLight.position.set(this.currentRootPos.x + 5, 8, this.currentRootPos.z + 3);
        this.keyLight.target.position.set(this.currentRootPos.x, 0, this.currentRootPos.z);
        this.keyLight.target.updateMatrixWorld();
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    resize() {
        const container = document.getElementById('canvas-container');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.motionApp = new MotionApp();
});
