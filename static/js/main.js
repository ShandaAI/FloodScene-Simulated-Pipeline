class FloodSceneApp {
    constructor() {
        this.sessionId = null;
        this.motionSocket = null;
        this.audioSocket = null;
        this.videoSocket = null;
        this.audioTimer = null;
        this.videoTimer = null;
        this.frameCount = 0;
        this.fpsCounter = 0;
        this.fpsUpdatedAt = performance.now();
        this.lastRoot = new THREE.Vector3(0, 1, 0);

        this.bindElements();
        this.initScene();
        this.bindEvents();
        this.loadConfig();
        this.animate();
        if (window.lucide) window.lucide.createIcons();
    }

    bindElements() {
        this.promptInput = document.getElementById('promptInput');
        this.audioToggle = document.getElementById('audioToggle');
        this.videoToggle = document.getElementById('videoToggle');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.updateBtn = document.getElementById('updateBtn');
        this.frameCountEl = document.getElementById('frameCount');
        this.fpsValue = document.getElementById('fpsValue');
        this.budgetValue = document.getElementById('budgetValue');
        this.sessionValue = document.getElementById('sessionValue');
        this.asrState = document.getElementById('asrState');
        this.vlmState = document.getElementById('vlmState');
        this.ttsState = document.getElementById('ttsState');
        this.motionState = document.getElementById('motionState');
        this.transcriptBox = document.getElementById('transcriptBox');
        this.sceneBox = document.getElementById('sceneBox');
        this.eventLog = document.getElementById('eventLog');
        this.connectionState = document.getElementById('connectionState');
        this.stageTitle = document.getElementById('stageTitle');
    }

    initScene() {
        const canvas = document.getElementById('renderCanvas');
        const container = document.getElementById('canvas-container');
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xe5e8ec);
        this.scene.fog = new THREE.Fog(0xe5e8ec, 14, 42);

        this.camera = new THREE.PerspectiveCamera(
            48,
            container.clientWidth / container.clientHeight,
            0.1,
            120,
        );
        this.camera.position.set(4.8, 2.3, 5.0);
        this.camera.lookAt(0, 0.8, 0);

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.08;

        this.scene.add(new THREE.HemisphereLight(0xdbeafe, 0x3a332f, 0.75));
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.22));

        this.keyLight = new THREE.DirectionalLight(0xfff2dc, 1.25);
        this.keyLight.position.set(4, 8, 5);
        this.keyLight.castShadow = true;
        this.keyLight.shadow.mapSize.set(2048, 2048);
        this.keyLight.shadow.camera.left = -8;
        this.keyLight.shadow.camera.right = 8;
        this.keyLight.shadow.camera.top = 8;
        this.keyLight.shadow.camera.bottom = -8;
        this.scene.add(this.keyLight);
        this.scene.add(this.keyLight.target);

        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(18, 13),
            new THREE.MeshStandardMaterial({ color: 0xbfc7c4, roughness: 0.82 }),
        );
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);

        const grid = new THREE.GridHelper(18, 18, 0x63706b, 0xd7ddda);
        grid.position.y = 0.012;
        this.scene.add(grid);

        const targetRing = new THREE.Mesh(
            new THREE.RingGeometry(1.6, 1.62, 96),
            new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 }),
        );
        targetRing.rotation.x = -Math.PI / 2;
        targetRing.position.y = 0.018;
        this.scene.add(targetRing);

        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 0.9, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.07;

        this.skeleton = new Skeleton3D(this.scene);
        window.addEventListener('resize', () => this.resize());
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.updateBtn.addEventListener('click', () => this.updatePrompt());
    }

    async loadConfig() {
        const response = await fetch('/api/config');
        const config = await response.json();
        this.budgetValue.textContent = `${Math.round(config.budget_remaining_seconds)}s`;
    }

    wsUrl(path) {
        const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${scheme}//${window.location.host}${path}`;
    }

    async start() {
        this.setButtons(true);
        this.log('Session', 'creating HF Endpoint simulation session');
        const response = await fetch('/api/session/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: this.promptInput.value,
                audio_enabled: this.audioToggle.checked,
                video_enabled: this.videoToggle.checked,
                endpoint_mode: 'hf-endpoint-sim',
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            this.log('Budget', error.detail || 'session start failed');
            this.setButtons(false);
            return;
        }

        const data = await response.json();
        this.sessionId = data.session_id;
        this.sessionValue.textContent = this.sessionId.slice(0, 6);
        this.transcriptBox.textContent = data.transcript || 'audio stream disabled';
        this.sceneBox.textContent = data.scene_context;
        this.budgetValue.textContent = `${Math.round(data.budget_remaining_seconds)}s`;
        this.frameCount = 0;
        this.fpsCounter = 0;
        this.skeleton.clearTrail();
        this.stageTitle.textContent = 'Streaming motion endpoint';
        this.connectMotion();
        if (this.audioToggle.checked) this.connectAudio();
        if (this.videoToggle.checked) this.connectVideo();
    }

    connectMotion() {
        this.motionSocket = new WebSocket(this.wsUrl(`/ws/motion/${this.sessionId}`));
        this.connectionState.textContent = 'connecting';
        this.motionSocket.onopen = () => {
            this.connectionState.textContent = 'connected';
            this.motionState.textContent = 'endpoint streaming';
        };
        this.motionSocket.onmessage = event => {
            const data = JSON.parse(event.data);
            if (data.type === 'motion_frame') {
                this.motionState.textContent = 'endpoint streaming';
                this.skeleton.updatePose(data.joints);
                this.frameCount = data.frame_id;
                this.fpsCounter += 1;
                this.frameCountEl.textContent = this.frameCount;
                this.budgetValue.textContent = `${Math.round(data.budget_remaining_seconds)}s`;
                const root = data.joints[0];
                this.lastRoot.set(root[0], root[1], root[2]);
                this.updateFps();
            } else if (data.type === 'pipeline_event') {
                this.applyStage(data.stage);
                this.log(data.stage, data.detail);
            } else if (data.type === 'budget_exhausted') {
                this.log('Budget', data.detail);
                this.stop(false);
            }
        };
        this.motionSocket.onclose = () => {
            this.connectionState.textContent = 'disconnected';
            this.motionState.textContent = 'endpoint idle';
        };
    }

    connectAudio() {
        this.audioSocket = new WebSocket(this.wsUrl(`/ws/audio/${this.sessionId}`));
        let sequence = 0;
        this.audioSocket.onopen = () => {
            this.asrState.textContent = 'endpoint streaming';
            this.audioTimer = setInterval(() => {
                sequence += 1;
                const level = 0.45 + 0.38 * Math.sin(sequence * 0.38) + 0.1 * Math.sin(sequence * 0.91);
                this.audioSocket.send(JSON.stringify({
                    type: 'audio_chunk',
                    sequence,
                    level: Math.max(0, Math.min(1, level)),
                }));
            }, 250);
        };
        this.audioSocket.onmessage = event => {
            const data = JSON.parse(event.data);
            if (data.type === 'asr_partial') {
                this.transcriptBox.textContent = data.text;
                this.log('ASR', `partial transcript level ${data.level}`);
            }
        };
        this.audioSocket.onclose = () => {
            this.asrState.textContent = 'endpoint idle';
        };
    }

    connectVideo() {
        this.videoSocket = new WebSocket(this.wsUrl(`/ws/video/${this.sessionId}`));
        let sequence = 0;
        this.videoSocket.onopen = () => {
            this.vlmState.textContent = 'endpoint streaming';
            this.videoTimer = setInterval(() => {
                sequence += 1;
                const motionEnergy = 0.35 + 0.35 * Math.sin(sequence * 0.51 + 0.4);
                this.videoSocket.send(JSON.stringify({
                    type: 'video_keyframe',
                    sequence,
                    motion_energy: Math.max(0, Math.min(1, motionEnergy)),
                }));
            }, 850);
        };
        this.videoSocket.onmessage = event => {
            const data = JSON.parse(event.data);
            if (data.type === 'vlm_context') {
                this.sceneBox.textContent = data.context;
                this.log('VLM', `keyframe motion ${data.motion_energy}`);
            }
        };
        this.videoSocket.onclose = () => {
            this.vlmState.textContent = 'endpoint idle';
        };
    }

    async updatePrompt() {
        if (!this.sessionId) return;
        const response = await fetch(`/api/session/${this.sessionId}/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: this.promptInput.value }),
        });
        const data = await response.json();
        this.log('Motion', 'prompt updated');
        this.stageTitle.textContent = data.prompt.slice(0, 64);
    }

    async stop(callApi = true) {
        clearInterval(this.audioTimer);
        clearInterval(this.videoTimer);
        this.audioTimer = null;
        this.videoTimer = null;
        for (const socket of [this.motionSocket, this.audioSocket, this.videoSocket]) {
            if (socket && socket.readyState <= 1) socket.close();
        }
        if (callApi && this.sessionId) {
            await fetch(`/api/session/${this.sessionId}/reset`, { method: 'POST' });
        }
        this.sessionId = null;
        this.setButtons(false);
        this.connectionState.textContent = 'disconnected';
        this.stageTitle.textContent = 'Waiting for motion endpoint';
        this.sessionValue.textContent = 'idle';
        this.asrState.textContent = 'endpoint idle';
        this.vlmState.textContent = 'endpoint idle';
        this.ttsState.textContent = 'endpoint idle';
        this.motionState.textContent = 'endpoint idle';
    }

    setButtons(running) {
        this.startBtn.disabled = running;
        this.stopBtn.disabled = !running;
        this.updateBtn.disabled = !running;
    }

    applyStage(stage) {
        if (stage === 'ASR') this.asrState.textContent = 'endpoint ready';
        if (stage === 'VLM') this.vlmState.textContent = 'endpoint ready';
        if (stage === 'TTS') this.ttsState.textContent = 'endpoint ready';
        if (stage === 'Motion') this.motionState.textContent = 'endpoint ready';
    }

    log(stage, detail) {
        const row = document.createElement('div');
        row.className = 'event-row';
        row.innerHTML = `<span>${stage}</span><b>${detail}</b>`;
        this.eventLog.prepend(row);
        while (this.eventLog.children.length > 14) {
            this.eventLog.removeChild(this.eventLog.lastChild);
        }
    }

    updateFps() {
        const now = performance.now();
        if (now - this.fpsUpdatedAt > 1000) {
            this.fpsValue.textContent = this.fpsCounter;
            this.fpsCounter = 0;
            this.fpsUpdatedAt = now;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.target.lerp(new THREE.Vector3(this.lastRoot.x, 0.9, this.lastRoot.z), 0.035);
        this.keyLight.position.set(this.lastRoot.x + 4, 8, this.lastRoot.z + 5);
        this.keyLight.target.position.set(this.lastRoot.x, 0, this.lastRoot.z);
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
    window.floodSceneApp = new FloodSceneApp();
});
