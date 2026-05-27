class MotionApp {
    constructor() {
        this.sessionId = null;
        this.realtimeSocket = null;
        this.onlineEventSource = null;
        this.activeProvider = null;

        this.isRunning = false;
        this.inputMode = 'online';
        this.targetFps = 20;
        this.frameCount = 0;
        this.motionFpsCounter = 0;
        this.motionFpsUpdatedAt = performance.now();
        this.timerStartedAt = null;
        this.timerStoppedAt = null;
        this.latencyRecorded = false;
        this.latencySeconds = null;
        this.bufferCapacity = 4;
        this.smooth = 0.0;
        this.generationSeed = 11;
        this.cfgScale = 5.0;
        this.historyLength = 30;
        this.onlineBatchSize = 4;
        this.onlineApiBase = '/api/flooddiffusion';
        this.onlineFrameQueue = [];
        this.nextOnlineFrameAt = 0;
        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000;
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        this.displayFrameStartedAt = 0;
        this.sourceFrameIntervalMs = 1000 / 30;
        this.avatar = null;
        this.avatarKey = null;
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
        this.fpsEl = document.getElementById('fps');
        this.frameCountEl = document.getElementById('frameCount');
        this.latencyEl = document.getElementById('latency');
        this.motionText = document.getElementById('motionText');
        this.onlineModeBtn = document.getElementById('onlineModeBtn');
        this.offlineModeBtn = document.getElementById('offlineModeBtn');
        this.onlineInputPanel = document.getElementById('onlineInputPanel');
        this.offlineInputPanel = document.getElementById('offlineInputPanel');
        this.offlineScheduleRows = document.getElementById('offlineScheduleRows');
        this.addCueBtn = document.getElementById('addCueBtn');
        this.startResetBtn = document.getElementById('startResetBtn');
        this.configBtn = document.getElementById('configBtn');
        this.configModal = document.getElementById('configModal');
        this.modalSeed = document.getElementById('modalSeed');
        this.modalCfg = document.getElementById('modalCfg');
        this.modalHistory = document.getElementById('modalHistory');
        this.modalSmooth = document.getElementById('modalSmooth');
        this.modalSmoothValue = document.getElementById('modalSmoothValue');
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
        this.configBtn.addEventListener('click', () => this.openConfig());
        this.onlineModeBtn.addEventListener('click', () => this.setInputMode('online'));
        this.offlineModeBtn.addEventListener('click', () => this.setInputMode('offline'));
        this.addCueBtn.addEventListener('click', () => this.addOfflineCue());
        this.motionText.addEventListener('keydown', event => {
            if (event.key === 'Enter' && this.inputMode === 'online') {
                event.preventDefault();
                if (this.isRunning) this.updateOnlineText();
                else this.start();
            }
        });
        this.configDiscardBtn.addEventListener('click', () => this.closeConfig());
        this.configSaveBtn.addEventListener('click', () => this.saveConfig());
        this.modalSmooth.addEventListener('input', () => {
            this.modalSmoothValue.textContent = Number(this.modalSmooth.value).toFixed(2);
        });
        this.configModal.addEventListener('click', event => {
            if (event.target === this.configModal) this.closeConfig();
        });
        window.addEventListener('beforeunload', () => this.sendResetBeacon());
        this.renderOfflineSchedule();
        this.setInputMode('online');
    }

    setInputMode(mode) {
        if (this.isRunning) return;
        this.inputMode = mode;
        const isOnline = mode === 'online';
        this.onlineModeBtn.classList.toggle('active', isOnline);
        this.offlineModeBtn.classList.toggle('active', !isOnline);
        this.onlineInputPanel.hidden = !isOnline;
        this.offlineInputPanel.hidden = isOnline;
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
            this.onlineApiBase = config.online?.api_base_url || '/api/flooddiffusion';
            this.onlineBatchSize = Math.max(1, Number(config.online?.batch_size || 4));
            this.cfgScale = Number(config.online?.default_cfg ?? this.cfgScale);
            this.historyLength = Math.max(1, Number(config.online?.default_history_length || this.historyLength));
            if (Number.isInteger(config.kimodo_g1_default_seed)) {
                this.generationSeed = config.kimodo_g1_default_seed;
            }
            return config;
        } catch (error) {
            this.setStatus('Offline');
            throw error;
        }
    }

    async initAvatar(config) {
        const rendererName = config?.renderer || config?.visualization || 'g1';
        await this.ensureAvatar(rendererName);
    }

    async ensureAvatar(rendererName) {
        const avatarKey = String(rendererName || 'g1').toLowerCase();
        if (this.avatar && this.avatarKey === avatarKey) return;
        if (this.avatar) this.avatar.setVisible(false);
        this.avatar = AvatarFactory.create(avatarKey, this.scene);
        this.avatarKey = avatarKey;
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
            this.frameCount = 0;
            this.motionFpsCounter = 0;
            this.motionFpsUpdatedAt = performance.now();
            this.displayFramePrevious = null;
            this.displayFrameCurrent = null;
            this.updateFrameDisplay(0);
            this.updateBufferDisplay(0, this.bufferCapacity);
            this.updateLatencyDisplay();
            this.fpsEl.textContent = '0';

            if (this.inputMode === 'online') {
                await this.startOnline();
            } else {
                await this.startOffline();
            }
            this.setButtonState();
        } catch (error) {
            this.stopTimer();
            this.setStatus(error.message || 'Error');
            this.setControlsLocked(false);
        }
    }

    async startOnline() {
        const text = this.motionText.value.trim();
        if (!text) throw new Error('Online text cannot be empty.');

        await this.ensureAvatar('smplh');
        this.avatar.clearTrail();
        this.avatar.setVisible(false);
        this.onlineFrameQueue = [];
        this.updateBufferDisplay(0, this.onlineBatchSize);

        const response = await fetch(`${this.onlineApiBase}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                force: true,
                config: {
                    seed: this.generationSeed,
                    cfg: this.cfgScale,
                    max_history_length: this.historyLength,
                    smoothing_alpha: this.smooth,
                },
            }),
        });
        const data = await response.json();
        if (!response.ok || data.status !== 'success') {
            throw new Error(data.detail || data.message || 'FloodDiffusion session failed.');
        }

        this.sessionId = data.session_id;
        this.activeProvider = 'flooddiffusion';
        this.targetFps = Number(data.target_fps || 30);
        this.sourceFrameIntervalMs = 1000 / this.targetFps;
        this.nextOnlineFrameAt = performance.now() + this.sourceFrameIntervalMs;
        this.isRunning = true;
        this.connectOnlineStream();
    }

    async startOffline() {
        await this.ensureAvatar(this.config?.renderer || 'g1');
        this.avatar.clearTrail();
        this.avatar.setVisible(false);

        const sessionPayload = {
            renderer: this.config?.renderer || 'g1',
            config: {
                seed: this.generationSeed,
                smooth: this.smooth,
            },
            schedule: this.readOfflineSchedule(),
        };
        this.sessionId = 'local-offline';
        this.activeProvider = 'kimodo';
        this.isRunning = true;
        this.connectRealtime(sessionPayload);
    }

    connectOnlineStream() {
        if (this.onlineEventSource) this.onlineEventSource.close();
        const streamUrl = `${this.onlineApiBase}/sessions/${this.sessionId}/stream?batch_size=${this.onlineBatchSize}&realtime=1`;
        this.onlineEventSource = new EventSource(streamUrl);
        this.onlineEventSource.addEventListener('ready', () => {
            this.setStatus('Streaming');
        });
        this.onlineEventSource.addEventListener('motion', event => {
            this.enqueueOnlineMotionBatch(JSON.parse(event.data));
        });
        this.onlineEventSource.addEventListener('stopped', () => {
            if (this.activeProvider !== 'flooddiffusion') return;
            this.isRunning = false;
            this.sessionId = null;
            this.activeProvider = null;
            this.closeOnlineStream();
            this.stopTimer();
            this.setButtonState();
            this.setStatus('Stopped');
        });
        this.onlineEventSource.addEventListener('error', event => {
            console.error('FloodDiffusion stream error', event);
            if (this.activeProvider === 'flooddiffusion' && this.isRunning) {
                this.setStatus('Stream Error');
            }
        });
    }

    connectRealtime(sessionPayload) {
        const socket = new WebSocket(this.wsUrl('/api/offline'));
        socket.binaryType = 'arraybuffer';
        this.realtimeSocket = socket;
        socket.onopen = () => {
            if (this.realtimeSocket !== socket) return;
            socket.send(JSON.stringify(sessionPayload));
            this.setStatus('Generating');
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

    enqueueOnlineMotionBatch(data) {
        if (!data.rotmats_b64 || !data.transls) return;
        const bytes = this.decodeB64Bytes(data.rotmats_b64);
        const allRotmats = new Float32Array(bytes.buffer);
        const rotPerFrame = 22 * 9;
        const startFrameId = Number(data.seq || this.frameCount);

        for (let i = 0; i < data.count; i++) {
            const offset = i * rotPerFrame;
            const joints = new Float32Array(22 * 3);
            if (data.joints?.[i]) {
                for (let j = 0; j < 22; j++) {
                    joints[j * 3] = data.joints[i][j][0];
                    joints[j * 3 + 1] = data.joints[i][j][1];
                    joints[j * 3 + 2] = data.joints[i][j][2];
                }
            }
            const transl = data.transls[i] || [0, 0, 0];
            this.onlineFrameQueue.push({
                frameId: startFrameId + i + 1,
                root: data.joints?.[i]?.[0] || transl,
                bufferSize: this.onlineFrameQueue.length,
                bufferCapacity: this.onlineBatchSize,
                joints,
                rotmats: allRotmats.subarray(offset, offset + rotPerFrame),
                transl,
            });
        }
        if (this.onlineFrameQueue.length > this.onlineBatchSize * 3) {
            this.onlineFrameQueue.splice(0, this.onlineFrameQueue.length - this.onlineBatchSize * 3);
        }
        this.updateBufferDisplay(this.onlineFrameQueue.length, this.onlineBatchSize);
    }

    decodeB64Bytes(b64) {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return bytes;
    }

    consumeOnlineFrame(now) {
        if (!this.isRunning || this.activeProvider !== 'flooddiffusion') return;
        if (now < this.nextOnlineFrameAt || this.onlineFrameQueue.length === 0) return;

        this.nextOnlineFrameAt += this.sourceFrameIntervalMs;
        if (this.nextOnlineFrameAt < now) {
            this.nextOnlineFrameAt = now + this.sourceFrameIntervalMs;
        }

        const frame = this.onlineFrameQueue.shift();
        frame.bufferSize = this.onlineFrameQueue.length;
        frame.bufferCapacity = this.onlineBatchSize;
        this.enqueueDisplayFrame(frame);
        this.frameCount = frame.frameId;
        this.motionFpsCounter += 1;
        this.updateFrameDisplay(frame.frameId);
        this.updateBufferDisplay(this.onlineFrameQueue.length, this.onlineBatchSize);
        this.updateMotionFps();
        this.setStatus('Streaming');
    }

    async updateOnlineText() {
        if (this.activeProvider !== 'flooddiffusion' || !this.sessionId) return;
        const text = this.motionText.value.trim();
        if (!text) return;
        const response = await fetch(`${this.onlineApiBase}/sessions/${this.sessionId}/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.status !== 'success') {
            this.setStatus(data.detail || data.message || 'Update Error');
        }
    }

    applyRealtimeEvent(data) {
        if (data.type === 'session.started') {
            this.setButtonState();
            this.setStatus('Streaming');
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
        this.setStatus('Streaming');
    }

    copyFrame(frame) {
        const copied = {
            frameId: frame.frameId,
            root: [...frame.root],
            bufferSize: frame.bufferSize,
            bufferCapacity: frame.bufferCapacity,
        };
        if (frame.joints) copied.joints = new Float32Array(frame.joints);
        if (frame.rotations) copied.rotations = new Float32Array(frame.rotations);
        if (frame.vertices) copied.vertices = new Float32Array(frame.vertices);
        if (frame.rotmats) copied.rotmats = new Float32Array(frame.rotmats);
        if (frame.transl) copied.transl = [...frame.transl];
        return copied;
    }

    enqueueDisplayFrame(frame) {
        const copied = this.copyFrame(frame);
        if (!this.displayFrameCurrent) {
            this.displayFramePrevious = copied;
            this.displayFrameCurrent = copied;
            this.displayFrameStartedAt = performance.now();
            this.avatar.applyFrame(copied, 1);
            this.currentRootPos.set(copied.root[0], copied.root[1], copied.root[2]);
            this.recordLatency();
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

        const prev = this.displayFramePrevious;
        const curr = this.displayFrameCurrent;
        const alpha = Math.max(0, Math.min(1, (now - this.displayFrameStartedAt) / this.sourceFrameIntervalMs));
        const smooth = Math.max(0, Math.min(1, this.smooth));
        if (smooth <= 0) {
            this.avatar.applyFrame(curr, 1);
            this.currentRootPos.set(curr.root[0], curr.root[1], curr.root[2]);
            return;
        }
        const joints = new Float32Array(curr.joints.length);
        for (let i = 0; i < joints.length; i++) {
            const interpolated = prev.joints[i] * (1 - alpha) + curr.joints[i] * alpha;
            joints[i] = curr.joints[i] * (1 - smooth) + interpolated * smooth;
        }
        const interpolatedRoot = [
            prev.root[0] * (1 - alpha) + curr.root[0] * alpha,
            prev.root[1] * (1 - alpha) + curr.root[1] * alpha,
            prev.root[2] * (1 - alpha) + curr.root[2] * alpha,
        ];
        const root = [
            curr.root[0] * (1 - smooth) + interpolatedRoot[0] * smooth,
            curr.root[1] * (1 - smooth) + interpolatedRoot[1] * smooth,
            curr.root[2] * (1 - smooth) + interpolatedRoot[2] * smooth,
        ];
        this.avatar.applyFrame({ ...curr, root, joints }, 1);
        this.currentRootPos.set(root[0], root[1], root[2]);
    }

    async reset(callApi = true) {
        const sessionId = this.sessionId;
        const provider = this.activeProvider;
        this.isRunning = false;
        this.sessionId = null;
        this.activeProvider = null;
        this.closeSockets();
        if (callApi && provider === 'flooddiffusion' && sessionId) {
            await fetch(`${this.onlineApiBase}/sessions/${sessionId}`, { method: 'DELETE' }).catch(() => {});
        }
        this.clearTimer();
        this.setStatus('Idle');
        this.frameCount = 0;
        this.motionFpsCounter = 0;
        this.onlineFrameQueue = [];
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        if (this.avatar) {
            this.avatar.clearTrail();
            this.avatar.setVisible(false);
        }
        this.updateFrameDisplay(0);
        this.fpsEl.textContent = '0';
        this.updateBufferDisplay(0, this.bufferCapacity);
        this.updateLatencyDisplay();
        this.setButtonState();
    }

    sendResetBeacon() {
        if (this.activeProvider !== 'flooddiffusion' || !this.sessionId) return;
        fetch(`${this.onlineApiBase}/sessions/${this.sessionId}`, {
            method: 'DELETE',
            keepalive: true,
        }).catch(() => {});
    }

    closeSockets() {
        this.closeOnlineStream();
        if (this.realtimeSocket && this.realtimeSocket.readyState <= WebSocket.OPEN) {
            if (this.realtimeSocket.readyState === WebSocket.OPEN) {
                this.realtimeSocket.send(JSON.stringify({ type: 'session.close' }));
            }
            this.realtimeSocket.close();
        }
        this.realtimeSocket = null;
    }

    closeOnlineStream() {
        if (this.onlineEventSource) {
            this.onlineEventSource.close();
            this.onlineEventSource = null;
        }
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
        if (!this.bufferSizeEl) return;
        const bufferSize = Number.isFinite(size) ? size : Math.min(this.bufferCapacity, this.frameCount);
        const bufferCapacity = Number.isFinite(capacity) ? capacity : this.bufferCapacity;
        this.bufferCapacity = bufferCapacity;
        this.bufferSizeEl.textContent = `${Math.min(bufferSize, bufferCapacity)} / ${bufferCapacity}`;
    }

    openConfig() {
        this.modalSeed.value = String(this.generationSeed);
        this.modalCfg.value = String(this.cfgScale);
        this.modalHistory.value = String(this.historyLength);
        this.modalSmooth.value = String(this.smooth);
        this.modalSmoothValue.textContent = this.smooth.toFixed(2);
        this.configModal.hidden = false;
    }

    closeConfig() {
        this.configModal.hidden = true;
    }

    async saveConfig() {
        const nextSeed = Math.max(0, Math.min(2147483647, Math.floor(Number(this.modalSeed.value) || 0)));
        const nextCfg = Math.max(0, Number(this.modalCfg.value) || 0);
        const nextHistory = Math.max(1, Math.floor(Number(this.modalHistory.value) || 1));
        const nextSmooth = Math.max(0, Math.min(1, Number(this.modalSmooth.value)));
        const shouldRestart = this.isRunning;
        this.generationSeed = nextSeed;
        this.cfgScale = nextCfg;
        this.historyLength = nextHistory;
        this.smooth = nextSmooth;
        this.closeConfig();
        if (shouldRestart) {
            await this.reset();
            await this.start();
        }
    }

    setStatus(status) {
        this.statusEl.textContent = status;
    }

    startTimer() {
        this.timerStartedAt = performance.now();
        this.timerStoppedAt = null;
        this.latencyRecorded = false;
        this.latencySeconds = null;
        this.updateLatencyDisplay();
    }

    stopTimer() {
        if (this.timerStartedAt !== null && this.timerStoppedAt === null) {
            this.timerStoppedAt = performance.now();
        }
    }

    clearTimer() {
        this.timerStartedAt = null;
        this.timerStoppedAt = null;
        this.latencyRecorded = false;
        this.latencySeconds = null;
        this.updateLatencyDisplay();
    }

    recordLatency() {
        if (this.latencyRecorded || this.timerStartedAt === null) return;
        this.latencySeconds = Math.max(0, (performance.now() - this.timerStartedAt) / 1000);
        this.latencyRecorded = true;
        this.updateLatencyDisplay();
    }

    updateLatencyDisplay() {
        if (!this.latencyEl) return;
        if (this.timerStartedAt === null) {
            this.latencyEl.textContent = '0.00s';
            return;
        }
        if (!this.latencyRecorded || this.latencySeconds === null) {
            this.latencyEl.textContent = '...';
            return;
        }
        this.latencyEl.textContent = `${this.latencySeconds.toFixed(2)}s`;
    }

    setControlsLocked(locked) {
        this.startResetBtn.disabled = locked;
        this.configBtn.disabled = false;
    }

    setButtonState() {
        this.startResetBtn.disabled = false;
        this.startResetBtn.textContent = this.isRunning ? 'Reset' : 'Start';
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
        this.consumeOnlineFrame(now);
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
