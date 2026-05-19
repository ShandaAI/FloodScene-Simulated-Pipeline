class Skeleton3D {
    constructor(scene) {
        this.scene = scene;
        this.joints = [];
        this.bones = [];
        this.trailPoints = [];
        this.maxTrailPoints = 180;

        this.chains = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        ];

        this.boneConnections = [];
        for (const chain of this.chains) {
            for (let i = 0; i < chain.length - 1; i++) {
                this.boneConnections.push([chain[i], chain[i + 1]]);
            }
        }

        this.jointMaterial = new THREE.MeshStandardMaterial({
            color: 0x1f9fb6,
            metalness: 0.15,
            roughness: 0.42,
            emissive: 0x042833,
            emissiveIntensity: 0.2,
        });

        const colors = [0xf2b84b, 0x1f9fb6, 0x314f8f, 0xd85656, 0x47a56d];
        this.boneMaterials = colors.map(color => new THREE.MeshStandardMaterial({
            color,
            metalness: 0.12,
            roughness: 0.45,
            emissive: color,
            emissiveIntensity: 0.04,
        }));

        this.initSkeleton();
        this.initTrail();
    }

    initSkeleton() {
        const jointGeometry = new THREE.SphereGeometry(0.035, 18, 18);
        for (let i = 0; i < 22; i++) {
            const joint = new THREE.Mesh(jointGeometry, this.jointMaterial);
            joint.castShadow = true;
            joint.receiveShadow = true;
            this.joints.push(joint);
            this.scene.add(joint);
        }

        for (let chainIndex = 0; chainIndex < this.chains.length; chainIndex++) {
            const material = this.boneMaterials[chainIndex];
            for (let i = 0; i < this.chains[chainIndex].length - 1; i++) {
                const bone = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.018, 0.018, 1, 10),
                    material,
                );
                bone.castShadow = true;
                bone.receiveShadow = true;
                this.bones.push(bone);
                this.scene.add(bone);
            }
        }
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
        this.scene.add(this.trail);
    }

    updatePose(jointPositions) {
        if (!jointPositions || jointPositions.length !== 22) return;

        for (let i = 0; i < 22; i++) {
            const p = jointPositions[i];
            this.joints[i].position.set(p[0], p[1], p[2]);
        }

        for (let i = 0; i < this.boneConnections.length; i++) {
            const [startIndex, endIndex] = this.boneConnections[i];
            const start = new THREE.Vector3(...jointPositions[startIndex]);
            const end = new THREE.Vector3(...jointPositions[endIndex]);
            this.updateBone(this.bones[i], start, end);
        }

        this.updateTrail(jointPositions[0]);
    }

    updateBone(bone, start, end) {
        const direction = new THREE.Vector3().subVectors(end, start);
        const length = direction.length();
        if (length < 0.001) return;
        bone.position.copy(new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5));
        bone.scale.y = length;
        bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
    }

    updateTrail(root) {
        const next = { x: root[0], y: 0.025, z: root[2] };
        const prev = this.trailPoints[this.trailPoints.length - 1];
        if (!prev || Math.hypot(next.x - prev.x, next.z - prev.z) > 0.025) {
            this.trailPoints.push(next);
        }
        if (this.trailPoints.length > this.maxTrailPoints) {
            this.trailPoints.shift();
        }

        const positions = this.trailGeometry.attributes.position.array;
        const colors = this.trailGeometry.attributes.color.array;
        for (let i = 0; i < this.trailPoints.length; i++) {
            const point = this.trailPoints[i];
            const alpha = i / Math.max(1, this.trailPoints.length - 1);
            positions[i * 3] = point.x;
            positions[i * 3 + 1] = point.y;
            positions[i * 3 + 2] = point.z;
            colors[i * 3] = 0.12 + 0.6 * alpha;
            colors[i * 3 + 1] = 0.75;
            colors[i * 3 + 2] = 0.9 - 0.25 * alpha;
        }
        this.trailGeometry.attributes.position.needsUpdate = true;
        this.trailGeometry.attributes.color.needsUpdate = true;
        this.trailGeometry.setDrawRange(0, this.trailPoints.length);
    }

    clearTrail() {
        this.trailPoints = [];
        this.trailGeometry.setDrawRange(0, 0);
    }
}

window.Skeleton3D = Skeleton3D;
