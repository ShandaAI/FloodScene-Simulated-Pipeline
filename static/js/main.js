class MotionApp {
    constructor() {
        this.sessionId = null;
        this.realtimeSocket = null;

        this.isRunning = false;
        this.inputMode = 'online';
        this.targetFps = 20;
        this.frameCount = 0;
        this.motionFpsCounter = 0;
        this.motionFpsUpdatedAt = performance.now();
        this.motionFpsFirstFrameAt = null;
        this.timerStartedAt = null;
        this.timerStoppedAt = null;
        this.latencyRecorded = false;
        this.latencySeconds = null;
        this.bufferCapacity = 4;
        this.smooth = 0.0;
        this.generationSeed = 11;
        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000;
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        this.displayFrameStartedAt = 0;
        this.appliedDisplayFrameId = null;
        this.firstFrameAppliedAt = null;
        this.sourceFrameIntervalMs = 1000 / 30;
        this.offlineAudioContext = null;
        this.offlineAudioPlayback = null;
        this.pageRecording = null;
        this.recordingUrl = null;
        this.recordingAudioDestination = null;
        this.hudCanvas = null;
        this.hudContext = null;
        this.hudTexture = null;
        this.hudScene = null;
        this.hudCamera = null;
        this.hudPlane = null;
        this.hudPlaneWidth = 0;
        this.hudPlaneHeight = 0;
        this.subtitleCanvas = null;
        this.subtitleContext = null;
        this.subtitleTexture = null;
        this.subtitlePlane = null;
        this.subtitlePlaneWidth = 0;
        this.subtitlePlaneHeight = 0;
        this.subtitleTimeline = [];
        this.subtitleWaveform = null;
        this.subtitleClockStartedAt = null;
        this.subtitleClockOffsetSeconds = 0;
        this.subtitleFadeSeconds = 0.16;
        this.avatar = null;
        this.avatarRenderer = null;
        this.meshReadyPromise = Promise.resolve();
        this.config = null;
        this.offlineCueRows = [
            { text: '江南style 跳舞', start: 0, end: '' },
            { text: '鞠躬', start: 8, end: 12 },
        ];
        this.offlineAudioRows = [
            { text: '', textStart: '', textEnd: '', audioFile: null, audioName: '' },
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
        this.audioStatusEl = document.getElementById('audioStatus');
        this.motionText = document.getElementById('motionText');
        this.onlineModeBtn = document.getElementById('onlineModeBtn');
        this.offlineModeBtn = document.getElementById('offlineModeBtn');
        this.offlineAudioModeBtn = document.getElementById('offlineAudioModeBtn');
        this.onlineInputPanel = document.getElementById('onlineInputPanel');
        this.offlineInputPanel = document.getElementById('offlineInputPanel');
        this.offlineAudioInputPanel = document.getElementById('offlineAudioInputPanel');
        this.offlineScheduleRows = document.getElementById('offlineScheduleRows');
        this.addCueBtn = document.getElementById('addCueBtn');
        this.offlineAudioRowsEl = document.getElementById('offlineAudioRows');
        this.addAudioCueBtn = document.getElementById('addAudioCueBtn');
        this.startResetBtn = document.getElementById('startResetBtn');
        this.recordToggle = document.getElementById('recordToggle');
        this.recordToggleLabel = this.recordToggle?.closest('.record-toggle') || null;
        this.recordingResult = document.getElementById('recordingResult');
        this.recordingStatus = document.getElementById('recordingStatus');
        this.recordingViewLink = document.getElementById('recordingViewLink');
        this.recordingDownloadLink = document.getElementById('recordingDownloadLink');
        this.configBtn = document.getElementById('configBtn');
        this.configModal = document.getElementById('configModal');
        this.modalSeed = document.getElementById('modalSeed');
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
        this.renderer.autoClear = false;
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

        this.initHud(container);
        window.addEventListener('resize', () => this.resize());
    }

    initHud(container) {
        this.hudCanvas = document.createElement('canvas');
        this.hudContext = this.hudCanvas.getContext('2d');
        this.hudTexture = new THREE.CanvasTexture(this.hudCanvas);
        this.hudTexture.minFilter = THREE.LinearFilter;
        this.hudTexture.magFilter = THREE.LinearFilter;

        const material = new THREE.MeshBasicMaterial({
            map: this.hudTexture,
            transparent: true,
            depthTest: false,
            depthWrite: false,
            toneMapped: false,
        });
        this.hudScene = new THREE.Scene();
        this.hudCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
        this.hudPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
        this.hudPlane.renderOrder = 1000;
        this.hudScene.add(this.hudPlane);

        this.subtitleCanvas = document.createElement('canvas');
        this.subtitleContext = this.subtitleCanvas.getContext('2d');
        this.subtitleTexture = new THREE.CanvasTexture(this.subtitleCanvas);
        this.subtitleTexture.minFilter = THREE.LinearFilter;
        this.subtitleTexture.magFilter = THREE.LinearFilter;
        const subtitleMaterial = new THREE.MeshBasicMaterial({
            map: this.subtitleTexture,
            transparent: true,
            depthTest: false,
            depthWrite: false,
            toneMapped: false,
        });
        this.subtitlePlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), subtitleMaterial);
        this.subtitlePlane.renderOrder = 1001;
        this.subtitlePlane.visible = false;
        this.hudScene.add(this.subtitlePlane);

        this.updateHudLayout(container);
    }

    updateHudLayout(container) {
        if (!this.hudCamera || !this.hudPlane) return;
        const width = Math.max(1, container.clientWidth);
        const height = Math.max(1, container.clientHeight);
        this.hudCamera.left = -width / 2;
        this.hudCamera.right = width / 2;
        this.hudCamera.top = height / 2;
        this.hudCamera.bottom = -height / 2;
        this.hudCamera.updateProjectionMatrix();

        this.hudPlaneWidth = Math.max(320, width - 48);
        this.hudPlaneHeight = 78;
        this.hudPlane.scale.set(this.hudPlaneWidth, this.hudPlaneHeight, 1);
        this.hudPlane.position.set(0, height / 2 - 18 - this.hudPlaneHeight / 2, 0);

        if (this.subtitlePlane) {
            this.subtitlePlaneWidth = Math.max(260, Math.min(width - 32, width * 0.82, 920));
            this.subtitlePlaneHeight = Math.max(72, Math.min(108, height * 0.16));
            this.subtitlePlane.scale.set(this.subtitlePlaneWidth, this.subtitlePlaneHeight, 1);
            this.subtitlePlane.position.set(0, -height / 2 + 22 + this.subtitlePlaneHeight / 2, 0);
        }
    }

    drawRoundedRect(context, x, y, width, height, radius) {
        const right = x + width;
        const bottom = y + height;
        context.beginPath();
        context.moveTo(x + radius, y);
        context.lineTo(right - radius, y);
        context.quadraticCurveTo(right, y, right, y + radius);
        context.lineTo(right, bottom - radius);
        context.quadraticCurveTo(right, bottom, right - radius, bottom);
        context.lineTo(x + radius, bottom);
        context.quadraticCurveTo(x, bottom, x, bottom - radius);
        context.lineTo(x, y + radius);
        context.quadraticCurveTo(x, y, x + radius, y);
        context.closePath();
    }

    drawHud() {
        if (!this.hudCanvas || !this.hudContext || !this.hudTexture) return;
        const dpr = 2;
        const width = Math.max(320, Math.round(this.hudPlaneWidth || 960));
        const height = Math.max(70, Math.round(this.hudPlaneHeight || 78));
        const canvasWidth = width * dpr;
        const canvasHeight = height * dpr;
        if (this.hudCanvas.width !== canvasWidth || this.hudCanvas.height !== canvasHeight) {
            this.hudCanvas.width = canvasWidth;
            this.hudCanvas.height = canvasHeight;
        }

        const context = this.hudContext;
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, canvasWidth, canvasHeight);
        context.setTransform(dpr, 0, 0, dpr, 0, 0);

        context.save();
        context.shadowColor = 'rgba(0, 0, 0, 0.05)';
        context.shadowBlur = 10;
        context.shadowOffsetY = 2;
        this.drawRoundedRect(context, 1, 1, width - 2, height - 2, 8);
        context.fillStyle = 'rgba(255, 255, 255, 0.78)';
        context.fill();
        context.restore();

        const items = [
            ['Status', this.statusEl?.textContent || 'Idle'],
            ['FPS', this.fpsEl?.textContent || '0'],
            ['Frames', this.frameCountEl?.textContent || '0'],
            ['Latency', this.latencyEl?.textContent || '0.00s'],
        ];
        const colWidth = width / items.length;

        items.forEach(([label, value], index) => {
            const x = index * colWidth + 16;
            context.fillStyle = 'rgba(95, 95, 95, 0.86)';
            context.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
            context.textBaseline = 'top';
            context.fillText(label.toUpperCase(), x, 14);
            context.fillStyle = 'rgba(5, 5, 5, 0.95)';
            context.font = '700 24px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
            context.fillText(String(value), x, 36, colWidth - 28);
        });

        this.hudTexture.needsUpdate = true;
    }

    normalizeSubtitleText(text) {
        return String(text || '').replace(/\s+/g, ' ').trim();
    }

    buildSubtitleTimeline(cues) {
        return cues
            .map(cue => ({
                text: this.normalizeSubtitleText(cue.text),
                start: Number(cue.start),
                end: Number(cue.end),
            }))
            .filter(cue => cue.text && Number.isFinite(cue.start) && Number.isFinite(cue.end) && cue.end > cue.start)
            .sort((a, b) => a.start - b.start);
    }

    buildOfflineSubtitleTimeline(schedule) {
        const cues = schedule.map((cue, index) => ({
            text: cue.text,
            start: cue.start,
            end: Number.isFinite(cue.end) ? cue.end : schedule[index + 1]?.start,
        }));
        return this.buildSubtitleTimeline(cues);
    }

    buildOfflineAudioSubtitleTimeline(inputs, buffers) {
        let cursor = 0;
        const cues = [];
        inputs.forEach((input, index) => {
            const duration = Number(buffers[index]?.duration);
            if (!Number.isFinite(duration) || duration <= 0) return;
            if (input.text) {
                const localStart = Math.max(0, Number(input.textStart || 0));
                const localEnd = input.textEnd === null ? duration : Number(input.textEnd);
                const start = cursor + Math.min(localStart, duration);
                const end = cursor + Math.min(Math.max(localEnd, localStart), duration);
                cues.push({ text: input.text, start, end });
            }
            cursor += duration;
        });
        return this.buildSubtitleTimeline(cues);
    }

    buildAudioWaveform(buffers) {
        const totalDuration = buffers.reduce((sum, item) => sum + Math.max(0, Number(item.duration) || 0), 0);
        if (!Number.isFinite(totalDuration) || totalDuration <= 0) return null;

        const binCount = Math.max(48, Math.min(260, Math.round(totalDuration * 30)));
        const values = [];
        for (let index = 0; index < binCount; index += 1) {
            const startTime = totalDuration * index / binCount;
            const endTime = totalDuration * (index + 1) / binCount;
            let cursor = 0;
            let squareSum = 0;
            let sampleCount = 0;

            for (const item of buffers) {
                const buffer = item.buffer;
                const duration = Math.max(0, Number(item.duration) || 0);
                const overlapStart = Math.max(startTime, cursor);
                const overlapEnd = Math.min(endTime, cursor + duration);
                if (!buffer || overlapEnd <= overlapStart) {
                    cursor += duration;
                    continue;
                }

                const localStart = overlapStart - cursor;
                const localEnd = overlapEnd - cursor;
                const sampleRate = buffer.sampleRate || 44100;
                const startFrame = Math.max(0, Math.floor(localStart * sampleRate));
                const endFrame = Math.min(buffer.length, Math.ceil(localEnd * sampleRate));
                const step = Math.max(1, Math.floor((endFrame - startFrame) / 90));
                const channelCount = Math.max(1, buffer.numberOfChannels || 1);

                for (let channel = 0; channel < channelCount; channel += 1) {
                    const data = buffer.getChannelData(channel);
                    for (let sample = startFrame; sample < endFrame; sample += step) {
                        const value = data[sample] || 0;
                        squareSum += value * value;
                        sampleCount += 1;
                    }
                }
                cursor += duration;
            }

            values.push(sampleCount ? Math.sqrt(squareSum / sampleCount) : 0);
        }

        const peak = Math.max(...values);
        const normalized = peak > 1e-5
            ? values.map(value => Math.max(0.08, Math.min(1, Math.sqrt(value / peak))))
            : values.map(() => 0.12);
        return { values: normalized, duration: totalDuration };
    }

    setSubtitleWaveform(waveform) {
        if (!waveform?.values?.length || !Number.isFinite(waveform.duration) || waveform.duration <= 0) {
            this.subtitleWaveform = null;
            return;
        }
        this.subtitleWaveform = waveform;
    }

    setSubtitleTimeline(timeline) {
        this.subtitleTimeline = this.buildSubtitleTimeline(timeline || []);
        this.subtitleClockStartedAt = null;
        this.subtitleClockOffsetSeconds = 0;
        if (this.subtitlePlane) this.subtitlePlane.visible = false;
    }

    startSubtitleClock(startedAt = performance.now(), offsetSeconds = 0) {
        if (!this.subtitleTimeline.length && !this.subtitleWaveform?.values?.length) return;
        this.subtitleClockStartedAt = startedAt;
        this.subtitleClockOffsetSeconds = Math.max(0, Number(offsetSeconds) || 0);
    }

    clearSubtitleTimeline() {
        this.subtitleTimeline = [];
        this.subtitleWaveform = null;
        this.subtitleClockStartedAt = null;
        this.subtitleClockOffsetSeconds = 0;
        if (this.subtitlePlane) this.subtitlePlane.visible = false;
    }

    getSubtitleTimeSeconds() {
        const playback = this.offlineAudioPlayback;
        if (playback?.scheduled && playback.startAudioTime !== null) {
            return Math.max(0, playback.context.currentTime - playback.startAudioTime);
        }
        if (this.subtitleClockStartedAt === null) return null;
        return Math.max(0, (performance.now() - this.subtitleClockStartedAt) / 1000 + this.subtitleClockOffsetSeconds);
    }

    getActiveSubtitle() {
        const time = this.getSubtitleTimeSeconds();
        if (time === null) return null;

        const fade = this.subtitleFadeSeconds;
        for (let index = this.subtitleTimeline.length - 1; index >= 0; index -= 1) {
            const cue = this.subtitleTimeline[index];
            if (time < cue.start || time > cue.end) continue;
            let opacity = 1;
            if (time - cue.start < fade) opacity = (time - cue.start) / fade;
            return { text: cue.text, opacity: Math.max(0, Math.min(1, opacity)) };
        }

        for (const cue of this.subtitleTimeline) {
            if (time < cue.start - fade || time > cue.end + fade) continue;
            let opacity = 0;
            if (time < cue.start) opacity = 1 - (cue.start - time) / fade;
            if (time > cue.end) opacity = 1 - (time - cue.end) / fade;
            return { text: cue.text, opacity: Math.max(0, Math.min(1, opacity)) };
        }
        return null;
    }

    getSubtitleWaveformValue(timeSeconds) {
        const waveform = this.subtitleWaveform;
        if (!waveform?.values?.length || !Number.isFinite(timeSeconds)) return 0;
        const duration = Math.max(0.001, waveform.duration);
        const clampedTime = Math.max(0, Math.min(duration, timeSeconds));
        const index = Math.max(0, Math.min(
            waveform.values.length - 1,
            Math.floor((clampedTime / duration) * waveform.values.length),
        ));
        return waveform.values[index] || 0;
    }

    getActiveSubtitleWaveform() {
        const waveform = this.subtitleWaveform;
        const time = this.getSubtitleTimeSeconds();
        if (!waveform?.values?.length || time === null) return null;

        const fade = this.subtitleFadeSeconds;
        if (time < -fade || time > waveform.duration + fade) return null;
        let opacity = 1;
        if (time < 0) opacity = 1 - Math.abs(time) / fade;
        if (time >= 0 && time < fade) opacity = Math.min(opacity, time / fade);
        if (time > waveform.duration) opacity = 1 - (time - waveform.duration) / fade;
        return {
            time: Math.max(0, Math.min(waveform.duration, time)),
            opacity: Math.max(0, Math.min(1, opacity)),
        };
    }

    drawSubtitleWaveform(context, x, centerY, width, height, timeSeconds) {
        const barCount = 15;
        const gap = width < 40 ? 2 : 3;
        const barWidth = Math.max(2, (width - gap * (barCount - 1)) / barCount);
        const windowSeconds = 0.9;
        context.save();
        context.strokeStyle = 'rgba(0, 0, 0, 0.78)';
        context.lineCap = 'round';
        context.lineWidth = Math.min(3, barWidth);

        for (let index = 0; index < barCount; index += 1) {
            const progress = barCount <= 1 ? 0.5 : index / (barCount - 1);
            const sampleTime = timeSeconds - windowSeconds / 2 + progress * windowSeconds;
            const value = this.getSubtitleWaveformValue(sampleTime);
            const barHeight = Math.max(5, value * height);
            const barX = x + index * (barWidth + gap) + barWidth / 2;
            context.beginPath();
            context.moveTo(barX, centerY - barHeight / 2);
            context.lineTo(barX, centerY + barHeight / 2);
            context.stroke();
        }
        context.restore();
    }

    fitSubtitleLine(context, text, maxWidth) {
        let line = text;
        while (line.length > 1 && context.measureText(`${line}...`).width > maxWidth) {
            line = line.slice(0, -1);
        }
        return `${line}...`;
    }

    wrapSubtitleText(context, text, maxWidth, maxLines = 2) {
        const normalized = this.normalizeSubtitleText(text);
        if (!normalized) return [];
        const tokenizedByWord = /\s/.test(normalized);
        const tokens = tokenizedByWord ? normalized.split(' ') : Array.from(normalized);
        const separator = tokenizedByWord ? ' ' : '';
        const lines = [];
        let line = '';
        let wasClipped = false;

        const pushLine = value => {
            const clean = value.trim();
            if (clean) lines.push(clean);
        };

        for (const token of tokens) {
            const candidate = line ? `${line}${separator}${token}` : token;
            if (context.measureText(candidate).width <= maxWidth) {
                line = candidate;
                continue;
            }
            if (line) pushLine(line);
            line = token;
            if (lines.length >= maxLines) {
                wasClipped = true;
                break;
            }

            while (context.measureText(line).width > maxWidth && Array.from(line).length > 1) {
                const chars = Array.from(line);
                let fitted = '';
                let consumed = 0;
                for (const char of chars) {
                    if (fitted && context.measureText(fitted + char).width > maxWidth) break;
                    fitted += char;
                    consumed += 1;
                }
                pushLine(fitted);
                line = chars.slice(consumed).join('');
                if (lines.length >= maxLines) {
                    wasClipped = true;
                    break;
                }
            }
        }

        if (lines.length < maxLines) {
            pushLine(line);
        } else if (line) {
            wasClipped = true;
        }

        const visibleLines = lines.slice(0, maxLines);
        if (wasClipped) visibleLines[maxLines - 1] = this.fitSubtitleLine(context, visibleLines[maxLines - 1], maxWidth);
        return visibleLines;
    }

    drawSubtitle() {
        if (!this.subtitleCanvas || !this.subtitleContext || !this.subtitleTexture || !this.subtitlePlane) return;

        const active = this.getActiveSubtitle();
        const waveform = this.getActiveSubtitleWaveform();
        const hasWaveform = Boolean(waveform && waveform.opacity > 0.01);
        const hasActiveText = Boolean(active && active.opacity > 0.01);
        if (!hasActiveText && !hasWaveform) {
            this.subtitlePlane.visible = false;
            return;
        }

        const dpr = 2;
        const width = Math.max(260, Math.round(this.subtitlePlaneWidth || 720));
        const height = Math.max(72, Math.round(this.subtitlePlaneHeight || 96));
        const canvasWidth = width * dpr;
        const canvasHeight = height * dpr;
        if (this.subtitleCanvas.width !== canvasWidth || this.subtitleCanvas.height !== canvasHeight) {
            this.subtitleCanvas.width = canvasWidth;
            this.subtitleCanvas.height = canvasHeight;
        }

        const context = this.subtitleContext;
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, canvasWidth, canvasHeight);
        context.setTransform(dpr, 0, 0, dpr, 0, 0);

        const fontSize = width < 420 ? 17 : 22;
        const lineHeight = Math.round(fontSize * 1.28);
        const paddingX = width < 420 ? 14 : 18;
        const waveformWidth = hasWaveform ? (width < 420 ? 54 : 74) : 0;
        const waveformHeight = hasWaveform ? (width < 420 ? 24 : 30) : 0;
        const waveformTextGap = hasWaveform && hasActiveText ? (width < 420 ? 8 : 10) : 0;
        context.font = `700 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
        context.textBaseline = 'top';

        const maxTextWidth = Math.max(80, width - paddingX * 2 - 18);
        const lines = hasActiveText ? this.wrapSubtitleText(context, active.text, maxTextWidth, 2) : [];
        const hasText = lines.length > 0;
        if (!hasText && !hasWaveform) {
            this.subtitlePlane.visible = false;
            context.globalAlpha = 1;
            return;
        }

        const textBlockHeight = hasText ? lines.length * lineHeight : 0;
        const contentHeight = hasText && hasWaveform
            ? waveformHeight + waveformTextGap + textBlockHeight
            : Math.max(waveformHeight, textBlockHeight);
        const contentTop = Math.round((height - contentHeight) / 2);
        if (hasWaveform) {
            context.globalAlpha = waveform.opacity;
            this.drawSubtitleWaveform(
                context,
                Math.round((width - waveformWidth) / 2),
                hasText ? contentTop + waveformHeight / 2 : height / 2,
                waveformWidth,
                waveformHeight,
                waveform.time,
            );
        }

        if (hasText) {
            context.globalAlpha = active.opacity;
            context.fillStyle = '#000000';
            const startY = hasWaveform
                ? contentTop + waveformHeight + waveformTextGap
                : Math.round((height - textBlockHeight) / 2);
            context.textAlign = 'center';
            lines.forEach((line, index) => {
                context.fillText(line, width / 2, startY + index * lineHeight, maxTextWidth);
            });
        }
        context.globalAlpha = 1;

        this.subtitlePlane.visible = true;
        this.subtitleTexture.needsUpdate = true;
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
        this.offlineAudioModeBtn.addEventListener('click', () => this.setInputMode('offline_audio'));
        this.addCueBtn.addEventListener('click', () => this.addOfflineCue());
        this.addAudioCueBtn.addEventListener('click', () => this.addOfflineAudioCue());
        this.motionText.addEventListener('keydown', event => {
            if (event.key === 'Enter' && this.inputMode === 'online') {
                event.preventDefault();
                if (!this.isRunning) this.start();
            }
        });
        this.configDiscardBtn.addEventListener('click', () => this.closeConfig());
        this.configSaveBtn.addEventListener('click', () => this.saveConfig());
        if (this.recordToggle) this.recordToggle.checked = false;
        this.modalSmooth.addEventListener('input', () => {
            this.modalSmoothValue.textContent = Number(this.modalSmooth.value).toFixed(2);
        });
        this.configModal.addEventListener('click', event => {
            if (event.target === this.configModal) this.closeConfig();
        });
        window.addEventListener('beforeunload', () => {
            this.stopPageRecording({ discard: true });
            this.sendResetBeacon();
        });
        this.renderOfflineSchedule();
        this.renderOfflineAudioSchedule();
        this.setInputMode('online');
    }

    setInputMode(mode) {
        if (this.isRunning) return;
        this.inputMode = mode;
        const isOnline = mode === 'online';
        const isOfflineText = mode === 'offline';
        const isOfflineAudio = mode === 'offline_audio';
        this.onlineModeBtn.classList.toggle('active', isOnline);
        this.offlineModeBtn.classList.toggle('active', isOfflineText);
        this.offlineAudioModeBtn.classList.toggle('active', isOfflineAudio);
        this.onlineInputPanel.hidden = !isOnline;
        this.offlineInputPanel.hidden = !isOfflineText;
        this.offlineAudioInputPanel.hidden = !isOfflineAudio;
        if (!isOfflineAudio) this.updateAudioStatus('Idle');
        this.clearSubtitleTimeline();
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

    escapeHtml(value) {
        return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
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

    renderOfflineAudioSchedule() {
        this.offlineAudioRowsEl.innerHTML = '';
        this.offlineAudioRows.forEach((cue, index) => {
            const row = document.createElement('div');
            row.className = 'offline-audio-row';
            row.dataset.index = String(index);
            row.innerHTML = `
                <input class="offline-audio-file" type="file" accept="audio/*">
                <input class="offline-audio-text" type="text" value="${this.escapeAttr(cue.text)}" placeholder="Optional text">
                <input class="offline-audio-text-start" type="number" min="0" step="0.1" value="${cue.textStart ?? ''}" placeholder="Text start">
                <input class="offline-audio-text-end" type="number" min="0" step="0.1" value="${cue.textEnd ?? ''}" placeholder="Text end">
                <span class="audio-file-name">${this.escapeHtml(cue.audioName || 'No audio')}</span>
                <button class="btn btn-compact remove-audio-cue" type="button"${this.offlineAudioRows.length <= 1 ? ' disabled' : ''}>Remove</button>
            `;
            row.querySelector('.offline-audio-file').addEventListener('change', event => {
                const file = event.target.files?.[0] || null;
                this.offlineAudioRows[index].audioFile = file;
                this.offlineAudioRows[index].audioName = file ? file.name : '';
                row.querySelector('.audio-file-name').textContent = file ? file.name : 'No audio';
            });
            row.querySelector('.remove-audio-cue').addEventListener('click', () => this.removeOfflineAudioCue(index));
            this.offlineAudioRowsEl.appendChild(row);
        });
    }

    addOfflineAudioCue() {
        this.offlineAudioRows.push({ text: '', textStart: '', textEnd: '', audioFile: null, audioName: '' });
        this.renderOfflineAudioSchedule();
    }

    removeOfflineAudioCue(index) {
        if (this.offlineAudioRows.length <= 1) return;
        this.offlineAudioRows.splice(index, 1);
        this.renderOfflineAudioSchedule();
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

    readFileAsDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ''));
            reader.onerror = () => reject(reader.error || new Error('Failed to read audio file.'));
            reader.readAsDataURL(file);
        });
    }

    collectOfflineAudioInputs() {
        const rows = [...this.offlineAudioRowsEl.querySelectorAll('.offline-audio-row')];
        if (!rows.length) throw new Error('Offline audio schedule is empty.');

        return rows.map(row => {
            const index = Number(row.dataset.index);
            const fileInput = row.querySelector('.offline-audio-file');
            const file = fileInput.files?.[0] || this.offlineAudioRows[index]?.audioFile;
            const text = row.querySelector('.offline-audio-text').value.trim();
            const textStartInput = row.querySelector('.offline-audio-text-start').value;
            const textEndInput = row.querySelector('.offline-audio-text-end').value;

            if (!file) throw new Error('Each offline audio cue needs an audio file.');

            const cue = {
                index,
                file,
                text,
                textStart: null,
                textEnd: null,
                textStartInput,
                textEndInput,
            };
            if (text) {
                const textStart = Number(textStartInput || 0);
                if (!Number.isFinite(textStart) || textStart < 0) throw new Error('Invalid text start.');
                cue.textStart = textStart;
                if (textEndInput !== '') {
                    const textEnd = Number(textEndInput);
                    if (!Number.isFinite(textEnd) || textEnd <= textStart) {
                        throw new Error('Text end must be after text start.');
                    }
                    cue.textEnd = textEnd;
                }
            }
            return cue;
        });
    }

    async readOfflineAudioSchedule(inputs = null) {
        const audioInputs = inputs || this.collectOfflineAudioInputs();
        const schedule = [];
        for (const input of audioInputs) {
            const file = input.file;
            const cue = {
                audio: {
                    name: file.name || `audio-${input.index}.wav`,
                    mime_type: file.type || 'application/octet-stream',
                    data: await this.readFileAsDataUrl(file),
                },
            };
            if (input.text) {
                cue.text = input.text;
                cue.text_start = input.textStart;
                if (input.textEnd !== null) cue.text_end = input.textEnd;
            }
            schedule.push(cue);
            this.offlineAudioRows[input.index] = {
                text: input.text,
                textStart: input.text ? input.textStart : '',
                textEnd: input.text ? input.textEndInput : '',
                audioFile: file,
                audioName: file.name,
            };
        }
        return schedule;
    }

    ensureOfflineAudioContext() {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error('This browser does not support local audio playback.');
        if (!this.offlineAudioContext) this.offlineAudioContext = new AudioContextClass();
        return this.offlineAudioContext;
    }

    async unlockOfflineAudioContext() {
        const context = this.ensureOfflineAudioContext();
        if (context.state === 'suspended') await context.resume();
        return context;
    }

    async prepareOfflineAudioPlayback(inputs) {
        this.stopOfflineAudioPlayback(false);
        if (!inputs || !inputs.length) return;
        const context = await this.unlockOfflineAudioContext();
        const buffers = [];
        for (const input of inputs) {
            const arrayBuffer = await input.file.arrayBuffer();
            const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
            buffers.push({
                name: input.file.name || `audio-${input.index}.wav`,
                buffer,
                duration: buffer.duration,
            });
        }
        this.offlineAudioPlayback = {
            context,
            buffers,
            sources: [],
            endTimer: null,
            scheduled: false,
            completed: false,
            startAudioTime: null,
            startPerformanceTime: null,
        };
        this.setSubtitleWaveform(this.buildAudioWaveform(buffers));
        const totalSeconds = buffers.reduce((sum, item) => sum + item.duration, 0);
        this.updateAudioStatus(`Ready ${totalSeconds.toFixed(1)}s`);
        return this.offlineAudioPlayback;
    }

    startOfflineAudioPlaybackAtFirstFrame() {
        const playback = this.offlineAudioPlayback;
        if (!playback || playback.scheduled || !playback.buffers.length) return false;

        const schedule = () => {
            if (this.offlineAudioPlayback !== playback || playback.scheduled) return;
            const context = playback.context;
            const startAt = context.currentTime + 0.005;
            let cursor = startAt;
            playback.sources = playback.buffers.map(item => {
                const source = context.createBufferSource();
                source.buffer = item.buffer;
                source.connect(context.destination);
                if (this.pageRecording?.audioDestination?.context === context) {
                    source.connect(this.pageRecording.audioDestination);
                }
                source.start(cursor);
                cursor += item.duration;
                return source;
            });
            playback.scheduled = true;
            playback.completed = false;
            playback.startAudioTime = startAt;
            playback.startPerformanceTime = performance.now();
            this.startSubtitleClock(playback.startPerformanceTime);
            this.updateAudioStatus('Playing');
            const totalMs = Math.max(0, (cursor - startAt) * 1000);
            playback.endTimer = window.setTimeout(() => {
                if (this.offlineAudioPlayback !== playback) return;
                playback.completed = true;
                this.updateAudioStatus('Done');
            }, totalMs + 120);
        };

        if (playback.context.state !== 'running') {
            playback.context.resume().then(schedule).catch(() => {
                this.updateAudioStatus('Blocked');
            });
            return true;
        }
        schedule();
        return true;
    }

    stopOfflineAudioPlayback(updateStatus = true) {
        const playback = this.offlineAudioPlayback;
        if (playback) {
            playback.sources.forEach(source => {
                try {
                    source.stop();
                } catch (error) {
                    void error;
                }
            });
            if (playback.endTimer !== null) window.clearTimeout(playback.endTimer);
        }
        this.offlineAudioPlayback = null;
        if (updateStatus) this.updateAudioStatus('Idle');
    }

    getRecordingMimeType() {
        if (!window.MediaRecorder?.isTypeSupported) return '';
        const candidates = [
            'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
            'video/mp4;codecs=h264,aac',
            'video/mp4',
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=vp8,opus',
            'video/webm',
        ];
        return candidates.find(type => window.MediaRecorder.isTypeSupported(type)) || '';
    }

    recordingExtensionForMimeType(mimeType) {
        return String(mimeType || '').toLowerCase().includes('mp4') ? 'mp4' : 'webm';
    }

    buildRecordingName(mimeType) {
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        return `motion-recording-${stamp}.${this.recordingExtensionForMimeType(mimeType)}`;
    }

    formatRecordingSize(bytes) {
        if (!Number.isFinite(bytes) || bytes <= 0) return '';
        if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    clearRecordingResult() {
        if (this.recordingUrl) {
            URL.revokeObjectURL(this.recordingUrl);
            this.recordingUrl = null;
        }
        if (this.recordingResult) this.recordingResult.hidden = true;
        if (this.recordingStatus) this.recordingStatus.textContent = '';
        if (this.recordingViewLink) {
            this.recordingViewLink.hidden = true;
            this.recordingViewLink.href = '#';
        }
        if (this.recordingDownloadLink) {
            this.recordingDownloadLink.hidden = true;
            this.recordingDownloadLink.href = '#';
            this.recordingDownloadLink.download = 'motion-recording.mp4';
        }
    }

    showRecordingStatus(status) {
        if (!this.recordingResult || !this.recordingStatus) return;
        this.recordingResult.hidden = !status;
        this.recordingStatus.textContent = status;
        if (this.recordingViewLink) this.recordingViewLink.hidden = true;
        if (this.recordingDownloadLink) this.recordingDownloadLink.hidden = true;
    }

    showRecordingLinks(blob, mimeType) {
        this.clearRecordingResult();
        const url = URL.createObjectURL(blob);
        const name = this.buildRecordingName(mimeType);
        this.recordingUrl = url;
        if (this.recordingResult) this.recordingResult.hidden = false;
        if (this.recordingStatus) {
            const size = this.formatRecordingSize(blob.size);
            this.recordingStatus.textContent = size ? `Saving ${size}` : 'Saving';
        }
        if (this.recordingViewLink) {
            this.recordingViewLink.href = url;
            this.recordingViewLink.type = mimeType || 'video/mp4';
            this.recordingViewLink.hidden = false;
        }
        if (this.recordingDownloadLink) {
            this.recordingDownloadLink.href = url;
            this.recordingDownloadLink.download = name;
            this.recordingDownloadLink.type = mimeType || 'video/mp4';
            this.recordingDownloadLink.hidden = false;
        }
    }

    updateRecordingLinks(url, filename, mimeType, status) {
        if (this.recordingResult) this.recordingResult.hidden = false;
        if (this.recordingStatus) this.recordingStatus.textContent = status;
        if (this.recordingViewLink) {
            this.recordingViewLink.href = url;
            this.recordingViewLink.type = mimeType || 'video/mp4';
            this.recordingViewLink.hidden = false;
        }
        if (this.recordingDownloadLink) {
            this.recordingDownloadLink.href = url;
            this.recordingDownloadLink.download = filename;
            this.recordingDownloadLink.type = mimeType || 'video/mp4';
            this.recordingDownloadLink.hidden = false;
        }
    }

    async saveRecordingToServer(blob, mimeType) {
        const filename = this.buildRecordingName(mimeType);
        const query = new URLSearchParams({ filename }).toString();
        const response = await fetch(`/api/recordings?${query}`, {
            method: 'POST',
            headers: { 'Content-Type': mimeType || 'application/octet-stream' },
            body: blob,
        });
        if (!response.ok) throw new Error('Recording save failed.');
        const payload = await response.json();
        if (!payload?.url || !payload?.filename) throw new Error('Recording save response was invalid.');
        const size = this.formatRecordingSize(Number(payload.bytes));
        const extension = this.recordingExtensionForMimeType(payload.mime_type || payload.filename).toUpperCase();
        const status = size ? `Saved ${extension} ${size}` : `Saved ${extension}`;
        this.updateRecordingLinks(payload.url, payload.filename, payload.mime_type || mimeType, status);
    }

    ensureRecordingAudioDestination() {
        const context = this.ensureOfflineAudioContext();
        if (!this.recordingAudioDestination || this.recordingAudioDestination.context !== context) {
            this.recordingAudioDestination = context.createMediaStreamDestination();
        }
        return this.recordingAudioDestination;
    }

    createRecordingSilence(audioDestination) {
        if (!audioDestination) return null;
        const context = audioDestination.context;
        const gain = context.createGain();
        gain.gain.value = 0.000001;
        let source = null;
        if (context.createConstantSource) {
            source = context.createConstantSource();
            source.offset.value = 1;
        } else {
            source = context.createOscillator();
            source.frequency.value = 20;
        }
        source.connect(gain);
        gain.connect(audioDestination);
        source.start();
        return { source, gain };
    }

    stopRecordingSilence(silence) {
        if (!silence) return;
        try {
            silence.source.stop();
        } catch (error) {
            void error;
        }
        try {
            silence.source.disconnect();
            silence.gain.disconnect();
        } catch (error) {
            void error;
        }
    }

    createCanvasRecordingStream(includeAudio) {
        const canvas = this.renderer?.domElement || document.getElementById('renderCanvas');
        if (!canvas?.captureStream) throw new Error('Canvas recording is not supported in this browser.');
        const stream = canvas.captureStream(30);
        if (!stream.getVideoTracks().length) throw new Error('Recording needs a canvas video track.');

        let audioDestination = null;
        if (includeAudio) {
            audioDestination = this.ensureRecordingAudioDestination();
            audioDestination.stream.getAudioTracks().forEach(track => stream.addTrack(track));
        }
        return { stream, audioDestination };
    }

    async startPageRecording() {
        if (!window.MediaRecorder) throw new Error('Recording is not supported in this browser.');

        this.stopPageRecording({ discard: true });
        this.clearRecordingResult();
        this.showRecordingStatus('Recording');
        const { stream, audioDestination } = this.createCanvasRecordingStream(this.inputMode === 'offline_audio');
        const silentAudio = this.createRecordingSilence(audioDestination);

        const mimeType = this.getRecordingMimeType();
        let recorder = null;
        try {
            recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        } catch (error) {
            this.stopRecordingSilence(silentAudio);
            stream.getTracks().forEach(track => track.stop());
            this.clearRecordingResult();
            throw new Error('Recording could not start.');
        }

        const state = {
            recorder,
            stream,
            audioDestination,
            silentAudio,
            chunks: [],
            discard: false,
            finalized: false,
            started: true,
            startedAt: performance.now(),
        };
        this.pageRecording = state;

        recorder.addEventListener('dataavailable', event => {
            if (event.data && event.data.size > 0) state.chunks.push(event.data);
        });
        recorder.addEventListener('stop', () => this.finishPageRecording(state));
        recorder.addEventListener('error', () => {
            this.setStatus('Recording Error');
            this.stopPageRecording();
        });
        stream.getVideoTracks().forEach(track => {
            track.addEventListener('ended', () => {
                if (this.pageRecording === state) this.stopPageRecording();
            });
        });

        try {
            this.renderer.render(this.scene, this.camera);
            recorder.start(250);
        } catch (error) {
            state.discard = true;
            this.finishPageRecording(state);
            throw new Error('Recording could not start.');
        }
    }

    stopPageRecording(options = {}) {
        const state = this.pageRecording;
        if (!state || state.finalized) return;
        state.discard = state.discard || Boolean(options.discard);
        const recorder = state.recorder;
        if (recorder && recorder.state !== 'inactive') {
            try {
                if (recorder.state === 'recording') recorder.requestData();
            } catch (error) {
                void error;
            }
            try {
                recorder.stop();
                return;
            } catch (error) {
                void error;
            }
        }
        this.finishPageRecording(state);
    }

    finishPageRecording(state) {
        if (!state || state.finalized) return;
        state.finalized = true;
        this.stopRecordingSilence(state.silentAudio);
        state.stream.getTracks().forEach(track => track.stop());
        if (state.audioDestination && this.recordingAudioDestination === state.audioDestination) {
            this.recordingAudioDestination = null;
        }
        if (this.pageRecording === state) this.pageRecording = null;

        if (state.discard) {
            this.clearRecordingResult();
            return;
        }
        if (!state.chunks.length) {
            this.showRecordingStatus('No video');
            return;
        }
        const mimeType = state.recorder.mimeType || this.getRecordingMimeType() || 'video/webm';
        const blob = new Blob(state.chunks, { type: mimeType });
        this.showRecordingLinks(blob, mimeType);
        this.saveRecordingToServer(blob, mimeType).catch(() => {
            if (this.recordingStatus) {
                const size = this.formatRecordingSize(blob.size);
                this.recordingStatus.textContent = size ? `Saved local ${size}` : 'Saved local';
            }
        });
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
            return config;
        } catch (error) {
            this.setStatus('Offline');
            throw error;
        }
    }

    rendererKey(rendererName) {
        const normalized = String(rendererName || 'g1').toLowerCase();
        return normalized.includes('smplx') ? 'smplx' : 'g1';
    }

    async initAvatar(config) {
        await this.ensureAvatar(config?.renderer || config?.visualization || 'g1');
    }

    async ensureAvatar(rendererName) {
        const key = this.rendererKey(rendererName);
        if (this.avatar && this.avatarRenderer === key) return;
        if (this.avatar) {
            this.avatar.clearTrail();
            this.avatar.setVisible(false);
        }
        const avatar = AvatarFactory.create(key, this.scene);
        await avatar.loadTopology().catch(error => {
            this.setStatus(`${avatar.displayName} Missing`);
            throw error;
        });
        this.avatar = avatar;
        this.avatarRenderer = key;
    }

    wsUrl(path) {
        const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${scheme}//${window.location.host}${path}`;
    }

    async start() {
        this.setStatus('Loading');
        this.setControlsLocked(true);
        this.clearSubtitleTimeline();

        let offlineAudioInputs = null;
        let offlineTextSchedule = null;
        let onlineText = '';
        let audioUnlockPromise = Promise.resolve();
        let recordingStartPromise = Promise.resolve();
        try {
            if (this.inputMode === 'offline_audio') {
                offlineAudioInputs = this.collectOfflineAudioInputs();
            } else if (this.inputMode === 'offline') {
                offlineTextSchedule = this.readOfflineSchedule();
            } else {
                onlineText = this.motionText.value.trim();
            }

            if (this.recordToggle?.checked) {
                recordingStartPromise = this.startPageRecording();
            }
            if (this.inputMode === 'offline_audio') {
                this.updateAudioStatus('Preparing');
                audioUnlockPromise = this.unlockOfflineAudioContext();
            } else {
                this.stopOfflineAudioPlayback(false);
                this.updateAudioStatus('Idle');
            }
            await recordingStartPromise;
            this.startTimer();
            await this.meshReadyPromise;
            const sessionPayload = {
                renderer: this.config?.renderer || 'g1',
                input_mode: 'offline',
                config: {
                    seed: this.generationSeed,
                    smooth: this.smooth,
                },
            };
            if (this.inputMode === 'online') {
                sessionPayload.schedule = [{ text: onlineText, start: 0, end: 4 }];
                this.setSubtitleTimeline(sessionPayload.schedule);
            } else if (this.inputMode === 'offline_audio') {
                sessionPayload.renderer = 'smplx';
                sessionPayload.input_mode = 'offline_audio';
                await audioUnlockPromise;
                const [schedule, playback] = await Promise.all([
                    this.readOfflineAudioSchedule(offlineAudioInputs),
                    this.prepareOfflineAudioPlayback(offlineAudioInputs),
                ]);
                sessionPayload.schedule = schedule;
                this.setSubtitleTimeline(this.buildOfflineAudioSubtitleTimeline(offlineAudioInputs, playback?.buffers || []));
            } else {
                sessionPayload.schedule = offlineTextSchedule;
                this.setSubtitleTimeline(this.buildOfflineSubtitleTimeline(offlineTextSchedule));
            }

            await this.ensureAvatar(sessionPayload.renderer);
            if (!this.avatar) throw new Error('Avatar failed to initialize');
            this.frameCount = 0;
            this.motionFpsCounter = 0;
            this.motionFpsUpdatedAt = performance.now();
            this.motionFpsFirstFrameAt = null;
            this.displayFramePrevious = null;
            this.displayFrameCurrent = null;
            this.appliedDisplayFrameId = null;
            this.firstFrameAppliedAt = null;
            this.avatar.clearTrail();
            this.avatar.setVisible(false);
            this.updateFrameDisplay(0);
            this.updateBufferDisplay(0, this.bufferCapacity);
            this.updateLatencyDisplay();
            this.fpsEl.textContent = '0';

            this.sessionId = 'local-offline';
            this.isRunning = true;
            this.connectRealtime(sessionPayload);
            this.setButtonState();
        } catch (error) {
            this.stopTimer();
            this.stopOfflineAudioPlayback(true);
            this.stopPageRecording({ discard: true });
            this.clearSubtitleTimeline();
            this.setStatus(error.message || 'Error');
            this.setControlsLocked(false);
        }
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
                this.isRunning = false;
                this.sessionId = null;
                this.stopTimer();
                this.stopOfflineAudioPlayback(true);
                this.stopPageRecording();
                this.setButtonState();
                this.setStatus('Idle');
            }
        };
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
            this.stopPageRecording();
            this.setButtonState();
            this.setStatus('Complete');
        } else if (data.type === 'budget_exhausted') {
            this.stopTimer();
            this.stopOfflineAudioPlayback(true);
            this.stopPageRecording();
            this.setStatus('Budget');
            this.reset(false);
        } else if (data.type === 'error') {
            this.stopTimer();
            this.stopOfflineAudioPlayback(true);
            this.stopPageRecording();
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
        if (this.motionFpsFirstFrameAt === null) {
            this.motionFpsFirstFrameAt = performance.now();
            this.motionFpsUpdatedAt = this.motionFpsFirstFrameAt;
        }
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
        if (frame.rootOrient) copied.rootOrient = new Float32Array(frame.rootOrient);
        if (frame.poseBody) copied.poseBody = new Float32Array(frame.poseBody);
        if (frame.trans) copied.trans = new Float32Array(frame.trans);
        return copied;
    }

    markFirstDisplayedFrame(appliedAt) {
        if (this.firstFrameAppliedAt !== null) return;
        this.firstFrameAppliedAt = appliedAt;
        const subtitleStartedWithAudio = this.startOfflineAudioPlaybackAtFirstFrame();
        if (!subtitleStartedWithAudio) this.startSubtitleClock(appliedAt);
        this.recordLatency();
    }

    enqueueDisplayFrame(frame) {
        const copied = this.copyFrame(frame);
        if (!this.displayFrameCurrent) {
            const appliedAt = performance.now();
            this.displayFramePrevious = copied;
            this.displayFrameCurrent = copied;
            this.displayFrameStartedAt = appliedAt;
            this.avatar.applyFrame(copied, 1);
            this.appliedDisplayFrameId = copied.frameId;
            this.currentRootPos.set(copied.root[0], copied.root[1], copied.root[2]);
            this.markFirstDisplayedFrame(appliedAt);
            return;
        }
        this.displayFramePrevious = this.displayFrameCurrent;
        this.displayFrameCurrent = copied;
        this.displayFrameStartedAt = performance.now();
    }

    applyInterpolatedDisplayFrame(now) {
        if (!this.avatar || !this.displayFrameCurrent) return;
        if (!this.displayFramePrevious) {
            if (this.appliedDisplayFrameId !== this.displayFrameCurrent.frameId) {
                this.avatar.applyFrame(this.displayFrameCurrent, 1);
                this.appliedDisplayFrameId = this.displayFrameCurrent.frameId;
            }
            return;
        }

        const prev = this.displayFramePrevious;
        const curr = this.displayFrameCurrent;
        const alpha = Math.max(0, Math.min(1, (now - this.displayFrameStartedAt) / this.sourceFrameIntervalMs));
        const smooth = Math.max(0, Math.min(1, this.smooth));
        if (smooth <= 0) {
            if (this.appliedDisplayFrameId !== curr.frameId) {
                this.avatar.applyFrame(curr, 1);
                this.appliedDisplayFrameId = curr.frameId;
                this.currentRootPos.set(curr.root[0], curr.root[1], curr.root[2]);
            }
            return;
        }
        let joints = curr.joints;
        if (curr.joints && prev.joints && curr.joints.length === prev.joints.length) {
            joints = new Float32Array(curr.joints.length);
            for (let i = 0; i < joints.length; i++) {
                const interpolated = prev.joints[i] * (1 - alpha) + curr.joints[i] * alpha;
                joints[i] = curr.joints[i] * (1 - smooth) + interpolated * smooth;
            }
        }
        let vertices = curr.vertices;
        if (curr.vertices && prev.vertices && curr.vertices.length === prev.vertices.length) {
            vertices = new Float32Array(curr.vertices.length);
            for (let i = 0; i < vertices.length; i++) {
                const interpolated = prev.vertices[i] * (1 - alpha) + curr.vertices[i] * alpha;
                vertices[i] = curr.vertices[i] * (1 - smooth) + interpolated * smooth;
            }
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
        this.avatar.applyFrame({ ...curr, root, joints, vertices }, 1);
        this.currentRootPos.set(root[0], root[1], root[2]);
    }

    async reset(callApi = true) {
        const sessionId = this.sessionId;
        this.isRunning = false;
        this.sessionId = null;
        this.closeSockets();
        this.stopOfflineAudioPlayback(true);
        this.stopPageRecording();
        this.clearSubtitleTimeline();
        this.clearTimer();
        void callApi;
        void sessionId;
        this.setStatus('Idle');
        this.frameCount = 0;
        this.motionFpsCounter = 0;
        this.motionFpsFirstFrameAt = null;
        this.displayFramePrevious = null;
        this.displayFrameCurrent = null;
        this.appliedDisplayFrameId = null;
        this.firstFrameAppliedAt = null;
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
        return;
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
        if (this.motionFpsFirstFrameAt === null) return;
        if (now - this.motionFpsUpdatedAt > 250) {
            const elapsed = Math.max(0.001, (now - this.motionFpsFirstFrameAt) / 1000);
            this.fpsEl.textContent = String(Math.round(this.motionFpsCounter / elapsed));
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
        this.modalSmooth.value = String(this.smooth);
        this.modalSmoothValue.textContent = this.smooth.toFixed(2);
        this.configModal.hidden = false;
    }

    closeConfig() {
        this.configModal.hidden = true;
    }

    async saveConfig() {
        const nextSeed = Math.max(0, Math.min(2147483647, Math.floor(Number(this.modalSeed.value) || 0)));
        const nextSmooth = Math.max(0, Math.min(1, Number(this.modalSmooth.value)));
        const shouldRestart = this.isRunning;
        this.generationSeed = nextSeed;
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

    updateAudioStatus(status) {
        if (this.audioStatusEl) this.audioStatusEl.textContent = status;
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
        if (this.recordToggle) this.recordToggle.disabled = locked || this.isRunning;
        if (this.recordToggleLabel) this.recordToggleLabel.classList.toggle('disabled', locked || this.isRunning);
        this.configBtn.disabled = false;
    }

    setButtonState() {
        this.startResetBtn.disabled = false;
        this.startResetBtn.textContent = this.isRunning ? 'Reset' : 'Start';
        if (this.recordToggle) this.recordToggle.disabled = this.isRunning;
        if (this.recordToggleLabel) this.recordToggleLabel.classList.toggle('disabled', this.isRunning);
        this.onlineModeBtn.disabled = this.isRunning;
        this.offlineModeBtn.disabled = this.isRunning;
        this.offlineAudioModeBtn.disabled = this.isRunning;
        this.addCueBtn.disabled = this.isRunning;
        this.addAudioCueBtn.disabled = this.isRunning;
        this.offlineScheduleRows.querySelectorAll('input, button').forEach(element => {
            element.disabled = this.isRunning;
        });
        this.offlineAudioRowsEl.querySelectorAll('input, button').forEach(element => {
            element.disabled = this.isRunning;
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        const now = performance.now();
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
        this.renderer.clear();
        this.renderer.render(this.scene, this.camera);
        this.drawHud();
        this.drawSubtitle();
        this.renderer.clearDepth();
        this.renderer.render(this.hudScene, this.hudCamera);
    }

    resize() {
        const container = document.getElementById('canvas-container');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.updateHudLayout(container);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.motionApp = new MotionApp();
});
