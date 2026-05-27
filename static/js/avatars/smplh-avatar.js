class SMPLHAvatar extends BaseAvatar {
    constructor(scene) {
        super(scene);
        this.scene = scene;
        this.displayName = 'SMPL-H';
        this.mesh = null;
        this.geometry = null;
        this.positionAttribute = null;
        this.initialized = false;
        this.numVerts = 0;

        this.vTemplate = null;
        this.Jrest = null;
        this.parents = null;
        this.skinIdx = null;
        this.skinWt = null;
        this.lhRotmat = null;
        this.rhRotmat = null;

        this.globalRot = null;
        this.globalTrans = null;
        this.relBones = null;
        this.positions = null;
        this.trail = null;
        this.trailGeometry = null;
        this.trailPoints = [];
        this.maxTrailPoints = 720;
        this.trailFadeMs = 18000;

        this.material = new THREE.MeshStandardMaterial({
            color: 0xb38d73,
            roughness: 0.5,
            metalness: 0.05,
            side: THREE.DoubleSide,
        });
    }

    async loadTopology() {
        const response = await fetch('/api/flooddiffusion/smplh/static');
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || error.message || 'SMPL-H topology failed to load');
        }
        const data = await response.json();
        if (data.status !== 'success') {
            throw new Error(data.message || 'SMPL-H topology failed to load');
        }
        this.loadModelData(data);
        return data;
    }

    readFrame() {
        return null;
    }

    applyFrame(frame) {
        if (!frame || !frame.rotmats) return;
        const transl = frame.transl || frame.root || [0, 0, 0];
        this.skinFrame(frame.rotmats, transl);
        this.updateTrail(frame.root || transl);
    }

    setVisible(visible) {
        if (this.mesh) this.mesh.visible = visible;
        if (this.trail) this.trail.visible = visible && this.trailPoints.length > 1;
    }

    clearTrail() {
        this.trailPoints = [];
        if (this.trailGeometry) this.trailGeometry.setDrawRange(0, 0);
        if (this.trail) this.trail.visible = false;
        this.setVisible(false);
    }

    decodeB64Float32(b64) {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return new Float32Array(bytes.buffer);
    }

    decodeB64Int32(b64) {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return new Int32Array(bytes.buffer);
    }

    decodeB64Uint8(b64) {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return bytes;
    }

    loadModelData(data) {
        this.numVerts = data.num_vertices;
        this.vTemplate = this.decodeB64Float32(data.v_template_b64);
        this.Jrest = this.decodeB64Float32(data.J_rest_b64);
        this.parents = new Int32Array(data.parents);
        this.skinIdx = this.decodeB64Uint8(data.skin_indices_b64);
        this.skinWt = this.decodeB64Float32(data.skin_weights_b64);
        this.lhRotmat = this.decodeB64Float32(data.lh_rotmat_b64);
        this.rhRotmat = this.decodeB64Float32(data.rh_rotmat_b64);

        this.relBones = new Float64Array(52 * 3);
        for (let j = 0; j < 52; j++) {
            const p = this.parents[j];
            if (p >= 0 && p < 52) {
                this.relBones[j * 3] = this.Jrest[j * 3] - this.Jrest[p * 3];
                this.relBones[j * 3 + 1] = this.Jrest[j * 3 + 1] - this.Jrest[p * 3 + 1];
                this.relBones[j * 3 + 2] = this.Jrest[j * 3 + 2] - this.Jrest[p * 3 + 2];
            } else {
                this.relBones[j * 3] = this.Jrest[j * 3];
                this.relBones[j * 3 + 1] = this.Jrest[j * 3 + 1];
                this.relBones[j * 3 + 2] = this.Jrest[j * 3 + 2];
            }
        }

        this.globalRot = new Float64Array(52 * 9);
        this.globalTrans = new Float64Array(52 * 3);
        this.positions = new Float32Array(this.numVerts * 3);

        const faces = this.decodeB64Int32(data.faces_b64);
        this.geometry = new THREE.BufferGeometry();
        this.positionAttribute = new THREE.BufferAttribute(new Float32Array(this.numVerts * 3), 3);
        this.geometry.setAttribute('position', this.positionAttribute);
        this.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
        this.geometry.computeVertexNormals();

        this.mesh = new THREE.Mesh(this.geometry, this.material);
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;
        this.mesh.frustumCulled = false;
        this.mesh.visible = false;
        this.scene.add(this.mesh);
        this.initTrail();
        this.initialized = true;
    }

    initTrail() {
        this.trailGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.maxTrailPoints * 3);
        const colors = new Float32Array(this.maxTrailPoints * 3);
        this.trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.trailGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        this.trailGeometry.setDrawRange(0, 0);
        this.trail = new THREE.Line(this.trailGeometry, new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.88,
        }));
        this.trail.frustumCulled = false;
        this.trail.visible = false;
        this.scene.add(this.trail);
    }

    updateTrail(root) {
        if (!this.trailGeometry || !root) return;
        const now = performance.now();
        const next = { x: root[0], y: 0.026, z: root[2], t: now };
        const prev = this.trailPoints[this.trailPoints.length - 1];
        if (!prev || Math.hypot(next.x - prev.x, next.z - prev.z) > 0.04) {
            this.trailPoints.push(next);
        }
        const minTime = now - this.trailFadeMs;
        while (this.trailPoints.length && this.trailPoints[0].t < minTime) {
            this.trailPoints.shift();
        }
        if (this.trailPoints.length > this.maxTrailPoints) {
            this.trailPoints.splice(0, this.trailPoints.length - this.maxTrailPoints);
        }

        const positions = this.trailGeometry.attributes.position.array;
        const colors = this.trailGeometry.attributes.color.array;
        for (let i = 0; i < this.trailPoints.length; i++) {
            const point = this.trailPoints[i];
            const age = Math.max(0, Math.min(1, (now - point.t) / this.trailFadeMs));
            const strength = 1 - age;
            positions[i * 3] = point.x;
            positions[i * 3 + 1] = point.y;
            positions[i * 3 + 2] = point.z;
            colors[i * 3] = 0.78 - 0.62 * strength;
            colors[i * 3 + 1] = 0.9 - 0.5 * strength;
            colors[i * 3 + 2] = 1.0 - 0.22 * strength;
        }
        this.trailGeometry.attributes.position.needsUpdate = true;
        this.trailGeometry.attributes.color.needsUpdate = true;
        this.trailGeometry.setDrawRange(0, this.trailPoints.length);
        this.trail.visible = this.trailPoints.length > 1;
    }

    skinFrame(rotmats22, transl) {
        if (!this.initialized) return;

        const gR = this.globalRot;
        const par = this.parents;

        for (let i = 0; i < 22 * 9; i++) {
            gR[i] = rotmats22[i];
        }
        for (let i = 0; i < 15 * 9; i++) {
            gR[22 * 9 + i] = this.lhRotmat[i];
        }
        for (let i = 0; i < 15 * 9; i++) {
            gR[37 * 9 + i] = this.rhRotmat[i];
        }

        const gT = this.globalTrans;
        const bones = this.relBones;
        gT[0] = bones[0];
        gT[1] = bones[1];
        gT[2] = bones[2];

        for (let j = 1; j < 52; j++) {
            const p = par[j];
            if (p < 0 || p >= 52) continue;
            const pOff = p * 9;
            const jOff = j * 9;
            const pT = p * 3;
            const jT = j * 3;

            const l0 = gR[jOff];
            const l1 = gR[jOff + 1];
            const l2 = gR[jOff + 2];
            const l3 = gR[jOff + 3];
            const l4 = gR[jOff + 4];
            const l5 = gR[jOff + 5];
            const l6 = gR[jOff + 6];
            const l7 = gR[jOff + 7];
            const l8 = gR[jOff + 8];

            const p0 = gR[pOff];
            const p1 = gR[pOff + 1];
            const p2 = gR[pOff + 2];
            const p3 = gR[pOff + 3];
            const p4 = gR[pOff + 4];
            const p5 = gR[pOff + 5];
            const p6 = gR[pOff + 6];
            const p7 = gR[pOff + 7];
            const p8 = gR[pOff + 8];

            gR[jOff] = p0 * l0 + p1 * l3 + p2 * l6;
            gR[jOff + 1] = p0 * l1 + p1 * l4 + p2 * l7;
            gR[jOff + 2] = p0 * l2 + p1 * l5 + p2 * l8;
            gR[jOff + 3] = p3 * l0 + p4 * l3 + p5 * l6;
            gR[jOff + 4] = p3 * l1 + p4 * l4 + p5 * l7;
            gR[jOff + 5] = p3 * l2 + p4 * l5 + p5 * l8;
            gR[jOff + 6] = p6 * l0 + p7 * l3 + p8 * l6;
            gR[jOff + 7] = p6 * l1 + p7 * l4 + p8 * l7;
            gR[jOff + 8] = p6 * l2 + p7 * l5 + p8 * l8;

            const bx = bones[jT];
            const by = bones[jT + 1];
            const bz = bones[jT + 2];
            gT[jT] = p0 * bx + p1 * by + p2 * bz + gT[pT];
            gT[jT + 1] = p3 * bx + p4 * by + p5 * bz + gT[pT + 1];
            gT[jT + 2] = p6 * bx + p7 * by + p8 * bz + gT[pT + 2];
        }

        const vT = this.vTemplate;
        const J = this.Jrest;
        const sIdx = this.skinIdx;
        const sWt = this.skinWt;
        const pos = this.positions;
        const tx = transl[0];
        const ty = transl[1];
        const tz = transl[2];

        for (let v = 0; v < this.numVerts; v++) {
            const v3 = v * 3;
            const v4 = v * 4;
            const vx = vT[v3];
            const vy = vT[v3 + 1];
            const vz = vT[v3 + 2];
            let px = 0;
            let py = 0;
            let pz = 0;

            for (let k = 0; k < 4; k++) {
                const w = sWt[v4 + k];
                if (w < 1e-6) continue;
                const jIdx = sIdx[v4 + k];
                const r = jIdx * 9;
                const jt = jIdx * 3;
                const j3 = jIdx * 3;
                const dx = vx - J[j3];
                const dy = vy - J[j3 + 1];
                const dz = vz - J[j3 + 2];
                px += w * (gR[r] * dx + gR[r + 1] * dy + gR[r + 2] * dz + gT[jt]);
                py += w * (gR[r + 3] * dx + gR[r + 4] * dy + gR[r + 5] * dz + gT[jt + 1]);
                pz += w * (gR[r + 6] * dx + gR[r + 7] * dy + gR[r + 8] * dz + gT[jt + 2]);
            }

            pos[v3] = px + tx;
            pos[v3 + 1] = py + ty;
            pos[v3 + 2] = pz + tz;
        }

        this.positionAttribute.array.set(pos);
        this.positionAttribute.needsUpdate = true;
        this.geometry.computeVertexNormals();
        this.mesh.visible = true;
    }
}

window.SMPLHAvatar = SMPLHAvatar;
