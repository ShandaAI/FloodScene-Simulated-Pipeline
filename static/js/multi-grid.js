class MultiGridCell {
    constructor(app, element, index) {
        this.app = app;
        this.element = element;
        this.index = index;
        this.canvas = element.querySelector('.multi-canvas');
        this.statusEl = element.querySelector('.multi-cell-status');
        this.framesEl = element.querySelector('.multi-cell-frames');
        this.seedEl = element.querySelector('.multi-cell-seed');

        this.sessionId = null;
        this.socket = null;
        this.seed = 11 + index;
        this.workerIndex = index;
        this.frameCount = 0;
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        this.sourceFrameIntervalMs = 1000 / 30;
        this.topologyReady = false;
        this.frameBuffer = [];
        this.bufferReady = false;
        this.playing = false;
        this.playbackComplete = false;
        this.playbackStartedAt = 0;

        this.initScene();
        this.resize();
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.element);
    }

    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        this.camera = new THREE.PerspectiveCamera(44, 1, 0.1, 1000);
        this.camera.position.set(3.8, 1.9, 4.2);
        this.camera.lookAt(0, 0.85, 0);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.72));

        this.keyLight = new THREE.DirectionalLight(0xffffff, 0.86);
        this.keyLight.position.set(4.5, 7, 3.5);
        this.keyLight.castShadow = true;
        this.scene.add(this.keyLight);
        this.scene.add(this.keyLight.target);

        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(10, 10),
            new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.08 }),
        );
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);

        const grid = new THREE.GridHelper(10, 10, 0xd8d8d8, 0xeeeeee);
        grid.position.y = 0.012;
        this.scene.add(grid);

        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.target.set(0, 0.85, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;

        this.avatar = new G1Avatar(this.scene);
        this.avatar.showTrail = false;
    }

    async load() {
        if (this.topologyReady) return;
        this.setStatus('Loading');
        await this.avatar.loadTopology();
        this.avatar.setVisible(false);
        this.topologyReady = true;
        this.setStatus('Idle');
    }

    wsUrl(path) {
        const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${scheme}//${window.location.host}${path}`;
    }

    async start({ prompt, duration, seed, workerIndex }) {
        await this.reset();
        await this.load();

        this.seed = seed;
        this.workerIndex = workerIndex;
        this.seedEl.textContent = `Seed ${seed}`;
        this.setStatus('Creating');

        const response = await fetch('/api/realtime/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                renderer: 'g1',
                input_mode: 'offline',
                frame_rate: 30,
                seed,
                kimodo_worker_index: workerIndex,
                stream_realtime: false,
                charge_budget: false,
                schedule: [{ text: prompt, start: 0, end: duration }],
            }),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `Variant ${this.index + 1} failed to create`);
        }

        const data = await response.json();
        this.sessionId = data.session_id;
        this.connect();
    }

    connect() {
        const socket = new WebSocket(this.wsUrl(`/api/realtime/sessions/${this.sessionId}`));
        socket.binaryType = 'arraybuffer';
        this.socket = socket;

        socket.onopen = () => {
            if (this.socket !== socket) return;
            this.setStatus('Generating');
        };
        socket.onmessage = event => {
            if (this.socket !== socket) return;
            if (event.data instanceof ArrayBuffer) {
                this.applyBinaryMotionFrame(new Float32Array(event.data));
                return;
            }
            this.applyEvent(JSON.parse(event.data));
        };
        socket.onclose = () => {
            if (this.socket !== socket) return;
            this.socket = null;
            if (!this.bufferReady) this.markReady('Closed');
        };
    }

    applyEvent(event) {
        if (event.type === 'motion_generation.started') {
            this.setStatus('Generating');
        } else if (event.type === 'motion_generation.segment_completed') {
            this.setStatus('Receiving');
        } else if (event.type === 'offline_schedule.completed') {
            this.markReady('Ready');
        } else if (event.type === 'budget_exhausted') {
            this.markReady('Budget');
        } else if (event.type === 'error') {
            this.markReady(event.code || 'Error');
        }
    }

    applyBinaryMotionFrame(packet) {
        if (!this.avatar || this.bufferReady) return;
        const frame = this.avatar.readFrame(packet, 9);
        if (!frame) return;
        this.frameBuffer.push(this.copyFrame(frame));
        this.frameCount = this.frameBuffer.length;
        this.framesEl.textContent = `${this.frameCount} frames`;
        this.setStatus('Receiving');
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

    markReady(status) {
        if (this.bufferReady) return;
        this.bufferReady = true;
        this.playing = false;
        this.closeSocket(false);
        this.sessionId = null;
        this.setStatus(status);
        this.app.onCellReady();
    }

    startPlayback(startedAt) {
        this.playbackStartedAt = startedAt;
        this.playbackComplete = false;
        this.playing = this.frameBuffer.length > 0;

        if (!this.playing) {
            this.playbackComplete = true;
            this.setStatus('Empty');
            this.app.onCellPlaybackComplete();
            return;
        }

        this.avatar.setVisible(true);
        this.setStatus('Playing');
        this.applyPlaybackFrame(startedAt);
    }

    render(now) {
        this.applyPlaybackFrame(now);
        const target = new THREE.Vector3(this.currentRootPos.x, 0.85, this.currentRootPos.z);
        this.controls.target.lerp(target, 0.04);
        this.keyLight.position.set(this.currentRootPos.x + 4.5, 7, this.currentRootPos.z + 3.5);
        this.keyLight.target.position.set(this.currentRootPos.x, 0, this.currentRootPos.z);
        this.keyLight.target.updateMatrixWorld();
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    applyPlaybackFrame(now) {
        if (!this.playing || !this.avatar || this.frameBuffer.length === 0) return;

        const position = Math.max(0, (now - this.playbackStartedAt) / this.sourceFrameIntervalMs);
        const lastIndex = this.frameBuffer.length - 1;
        if (position >= this.frameBuffer.length) {
            this.applyFrameAt(lastIndex);
            this.playing = false;
            this.playbackComplete = true;
            this.setStatus('Complete');
            this.app.onCellPlaybackComplete();
            return;
        }
        if (position >= lastIndex) {
            this.applyFrameAt(lastIndex);
            return;
        }

        const previousIndex = Math.floor(position);
        const nextIndex = previousIndex + 1;
        const alpha = position - previousIndex;
        this.applyInterpolatedFrame(this.frameBuffer[previousIndex], this.frameBuffer[nextIndex], alpha);
    }

    applyFrameAt(index) {
        const frame = this.frameBuffer[index];
        this.avatar.applyFrame(frame, 1);
        this.currentRootPos.set(frame.root[0], frame.root[1], frame.root[2]);
    }

    applyInterpolatedFrame(previous, current, alpha) {
        const clampedAlpha = Math.max(0, Math.min(1, alpha));
        const joints = new Float32Array(current.joints.length);
        for (let i = 0; i < joints.length; i++) {
            joints[i] = previous.joints[i] * (1 - clampedAlpha) + current.joints[i] * clampedAlpha;
        }
        const root = [
            previous.root[0] * (1 - clampedAlpha) + current.root[0] * clampedAlpha,
            previous.root[1] * (1 - clampedAlpha) + current.root[1] * clampedAlpha,
            previous.root[2] * (1 - clampedAlpha) + current.root[2] * clampedAlpha,
        ];
        this.avatar.applyFrame({ ...current, root, joints }, 1);
        this.currentRootPos.set(root[0], root[1], root[2]);
    }

    async reset(callApi = true) {
        const sessionId = this.sessionId;
        this.closeSocket(true);
        this.sessionId = null;
        this.frameCount = 0;
        this.frameBuffer = [];
        this.bufferReady = false;
        this.playing = false;
        this.playbackComplete = false;
        this.playbackStartedAt = 0;
        this.framesEl.textContent = '0 frames';
        this.currentRootPos.set(0, 1, 0);
        if (this.avatar) {
            this.avatar.clearTrail();
            this.avatar.setVisible(false);
        }
        if (callApi && sessionId) {
            await fetch(`/api/realtime/sessions/${sessionId}/close`, { method: 'POST' }).catch(() => {});
        }
        this.setStatus(this.topologyReady ? 'Idle' : 'Loading');
    }

    closeSocket(sendClose) {
        if (this.socket && this.socket.readyState <= WebSocket.OPEN) {
            if (sendClose && this.socket.readyState === WebSocket.OPEN) {
                this.socket.send(JSON.stringify({ type: 'session.close' }));
            }
            this.socket.close();
        }
        this.socket = null;
    }

    resize() {
        const width = Math.max(1, this.canvas.clientWidth || this.element.getBoundingClientRect().width);
        const height = Math.max(160, Math.min(260, width * 0.58));
        this.canvas.style.height = `${height}px`;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height, false);
    }

    setStatus(status) {
        this.statusEl.textContent = status;
    }
}

class MultiGridApp {
    constructor() {
        this.cellCount = 8;
        this.cells = [];
        this.config = null;
        this.running = false;
        this.readyCells = 0;
        this.playbackCompleteCells = 0;
        this.playbackStarted = false;
        this.playbackTimer = null;
        this.timerStartedAt = null;
        this.timerStoppedAt = null;

        this.bindElements();
        this.cells = [...document.querySelectorAll('.multi-cell')].map((element, index) => (
            new MultiGridCell(this, element, index)
        ));
        this.init();
        this.animate();
    }

    bindElements() {
        this.statusEl = document.getElementById('multiStatus');
        this.completeEl = document.getElementById('multiComplete');
        this.elapsedEl = document.getElementById('multiElapsed');
        this.workersEl = document.getElementById('multiWorkers');
        this.promptEl = document.getElementById('multiPrompt');
        this.durationEl = document.getElementById('multiDuration');
        this.baseSeedEl = document.getElementById('multiBaseSeed');
        this.startResetBtn = document.getElementById('multiStartResetBtn');
        this.startResetBtn.addEventListener('click', () => {
            if (this.running) {
                this.reset();
            } else {
                this.start();
            }
        });
    }

    async init() {
        try {
            const response = await fetch('/api/config');
            this.config = await response.json();
            const defaultSeed = this.config.kimodo_g1_default_seed;
            if (Number.isInteger(defaultSeed)) this.baseSeedEl.value = String(defaultSeed);
            this.workersEl.textContent = String(this.config.kimodo_g1_worker_count || 0);
            this.updateCellSeeds();
            await Promise.all(this.cells.map(cell => cell.load()));
            this.setStatus('Idle');
        } catch (error) {
            this.setStatus('Offline');
        }
    }

    updateCellSeeds() {
        const baseSeed = this.readBaseSeed();
        this.cells.forEach((cell, index) => {
            cell.seed = baseSeed + index;
            cell.seedEl.textContent = `Seed ${cell.seed}`;
        });
    }

    readBaseSeed() {
        return Math.max(0, Math.min(2147483647, Math.floor(Number(this.baseSeedEl.value) || 0)));
    }

    readDuration() {
        return Math.max(1, Math.min(10, Number(this.durationEl.value) || 4));
    }

    async start() {
        const prompt = this.promptEl.value.trim();
        if (!prompt) {
            this.setStatus('Prompt required');
            return;
        }

        this.running = true;
        this.readyCells = 0;
        this.playbackCompleteCells = 0;
        this.playbackStarted = false;
        this.clearPlaybackTimer();
        this.updateReadyDisplay();
        this.setStatus('Starting');
        this.startResetBtn.textContent = 'Reset';
        this.startResetBtn.disabled = true;
        this.promptEl.disabled = true;
        this.durationEl.disabled = true;
        this.baseSeedEl.disabled = true;
        this.updateCellSeeds();
        this.startTimer();

        const baseSeed = this.readBaseSeed();
        const duration = this.readDuration();
        const starts = this.cells.map((cell, index) => cell.start({
            prompt,
            duration,
            seed: baseSeed + index,
            workerIndex: index,
        }));

        const results = await Promise.allSettled(starts);
        results.forEach((result, index) => {
            if (result.status !== 'rejected') return;
            const cell = this.cells[index];
            cell.markReady(result.reason?.message || 'Error');
        });
        this.startResetBtn.disabled = false;
        if (this.running && !this.playbackStarted) this.setStatus('Generating');
    }

    async reset() {
        this.running = false;
        this.readyCells = 0;
        this.playbackCompleteCells = 0;
        this.playbackStarted = false;
        this.clearPlaybackTimer();
        this.stopTimer();
        await Promise.all(this.cells.map(cell => cell.reset()));
        this.clearTimer();
        this.updateReadyDisplay();
        this.setStatus('Idle');
        this.startResetBtn.textContent = 'Start';
        this.startResetBtn.disabled = false;
        this.promptEl.disabled = false;
        this.durationEl.disabled = false;
        this.baseSeedEl.disabled = false;
    }

    onCellReady() {
        if (!this.running) return;
        this.readyCells = Math.min(this.cellCount, this.readyCells + 1);
        this.updateReadyDisplay();
        if (this.readyCells >= this.cellCount) {
            this.startSynchronizedPlayback();
        }
    }

    startSynchronizedPlayback() {
        if (!this.running || this.playbackStarted) return;
        this.playbackStarted = true;
        this.playbackCompleteCells = 0;
        this.setStatus('Ready');
        this.playbackTimer = window.setTimeout(() => {
            this.playbackTimer = null;
            if (!this.running) return;
            this.setStatus('Playing');
            const startedAt = performance.now() + 80;
            this.cells.forEach(cell => cell.startPlayback(startedAt));
        }, 350);
    }

    onCellPlaybackComplete() {
        if (!this.running) return;
        this.playbackCompleteCells = Math.min(this.cellCount, this.playbackCompleteCells + 1);
        if (this.playbackCompleteCells >= this.cellCount) {
            this.running = false;
            this.stopTimer();
            this.setStatus('Complete');
            this.startResetBtn.textContent = 'Start';
            this.promptEl.disabled = false;
            this.durationEl.disabled = false;
            this.baseSeedEl.disabled = false;
        }
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

    clearPlaybackTimer() {
        if (this.playbackTimer !== null) {
            window.clearTimeout(this.playbackTimer);
            this.playbackTimer = null;
        }
    }

    clearTimer() {
        this.timerStartedAt = null;
        this.timerStoppedAt = null;
        this.updateElapsedDisplay();
    }

    updateElapsedDisplay(now = performance.now()) {
        if (this.timerStartedAt === null) {
            this.elapsedEl.textContent = '0.00s';
            return;
        }
        const end = this.timerStoppedAt ?? now;
        this.elapsedEl.textContent = `${Math.max(0, (end - this.timerStartedAt) / 1000).toFixed(2)}s`;
    }

    updateReadyDisplay() {
        this.completeEl.textContent = `${this.readyCells} / ${this.cellCount}`;
    }

    setStatus(status) {
        this.statusEl.textContent = status;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        const now = performance.now();
        if (this.timerStartedAt !== null && this.timerStoppedAt === null) {
            this.updateElapsedDisplay(now);
        }
        this.cells.forEach(cell => cell.render(now));
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.multiGridApp = new MultiGridApp();
});
