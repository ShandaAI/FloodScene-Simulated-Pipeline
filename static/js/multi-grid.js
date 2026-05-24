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
        this.completed = false;
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        this.displayFrameStartedAt = 0;
        this.sourceFrameIntervalMs = 1000 / 30;
        this.ready = false;

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
        if (this.ready) return;
        this.setStatus('Loading');
        await this.avatar.loadTopology();
        this.avatar.setVisible(false);
        this.ready = true;
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
        this.completed = false;
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
            if (!this.completed) this.setStatus('Closed');
        };
    }

    applyEvent(event) {
        if (event.type === 'motion_generation.started') {
            this.setStatus('Generating');
        } else if (event.type === 'motion_generation.segment_completed') {
            this.setStatus('Streaming');
        } else if (event.type === 'offline_schedule.completed') {
            if (this.completed) return;
            this.completed = true;
            this.setStatus('Complete');
            this.closeSocket(false);
            this.app.onCellComplete();
        } else if (event.type === 'budget_exhausted') {
            if (this.completed) return;
            this.completed = true;
            this.setStatus('Budget');
            this.app.onCellComplete();
        } else if (event.type === 'error') {
            if (this.completed) return;
            this.completed = true;
            this.setStatus(event.code || 'Error');
            this.app.onCellComplete();
        }
    }

    applyBinaryMotionFrame(packet) {
        if (!this.avatar) return;
        const frame = this.avatar.readFrame(packet, 9);
        if (!frame) return;
        this.enqueueDisplayFrame(frame);
        this.frameCount = frame.frameId;
        this.framesEl.textContent = `${frame.frameId} frames`;
        this.setStatus('Streaming');
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

    render(now) {
        this.applyInterpolatedDisplayFrame(now);
        const target = new THREE.Vector3(this.currentRootPos.x, 0.85, this.currentRootPos.z);
        this.controls.target.lerp(target, 0.04);
        this.keyLight.position.set(this.currentRootPos.x + 4.5, 7, this.currentRootPos.z + 3.5);
        this.keyLight.target.position.set(this.currentRootPos.x, 0, this.currentRootPos.z);
        this.keyLight.target.updateMatrixWorld();
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
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

    async reset(callApi = true) {
        const sessionId = this.sessionId;
        this.closeSocket(true);
        this.sessionId = null;
        this.completed = false;
        this.frameCount = 0;
        this.framesEl.textContent = '0 frames';
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        this.currentRootPos.set(0, 1, 0);
        if (this.avatar) {
            this.avatar.clearTrail();
            this.avatar.setVisible(false);
        }
        if (callApi && sessionId) {
            await fetch(`/api/realtime/sessions/${sessionId}/close`, { method: 'POST' }).catch(() => {});
        }
        this.setStatus(this.ready ? 'Idle' : 'Loading');
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
        this.completedCells = 0;
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
        this.completedCells = 0;
        this.updateCompleteDisplay();
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
            cell.completed = true;
            cell.setStatus(result.reason?.message || 'Error');
            this.onCellComplete();
        });
        this.startResetBtn.disabled = false;
        if (this.running) this.setStatus('Streaming');
    }

    async reset() {
        this.running = false;
        this.completedCells = 0;
        this.stopTimer();
        await Promise.all(this.cells.map(cell => cell.reset()));
        this.clearTimer();
        this.updateCompleteDisplay();
        this.setStatus('Idle');
        this.startResetBtn.textContent = 'Start';
        this.startResetBtn.disabled = false;
        this.promptEl.disabled = false;
        this.durationEl.disabled = false;
        this.baseSeedEl.disabled = false;
    }

    onCellComplete() {
        if (!this.running) return;
        this.completedCells = Math.min(this.cellCount, this.completedCells + 1);
        this.updateCompleteDisplay();
        if (this.completedCells >= this.cellCount) {
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

    updateCompleteDisplay() {
        this.completeEl.textContent = `${this.completedCells} / ${this.cellCount}`;
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
