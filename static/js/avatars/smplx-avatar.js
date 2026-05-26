class SMPLXAvatar extends BaseAvatar {
    constructor(scene) {
        super(scene);
        this.scene = scene;
        this.displayName = 'SMPL-X';
        this.mesh = null;
        this.geometry = null;
        this.positionAttribute = null;
        this.vertexCount = 0;
        this.smplBones = [];
        this.smplSkeleton = null;
        this.restJoints = null;
        this.previousVertices = null;
        this.previousJoints = null;
        this.trailPoints = [];
        this.maxTrailPoints = 240;
        this.joints = [];
        this.bones = [];
        this.boneConnections = [];

        this.chains = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        ];

        this.material = new THREE.MeshStandardMaterial({
            color: 0xc99b7a,
            roughness: 0.66,
            metalness: 0.02,
            side: THREE.FrontSide,
            skinning: true,
        });

        this.initTrail();
        this.initSkeletonOverlay();
    }

    async loadTopology() {
        const response = await fetch('/api/smplx/topology');
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'SMPL-X topology failed to load');
        }

        const topology = await response.json();
        this.vertexCount = topology.vertex_count;
        const positions = topology.v_template ? new Float32Array(topology.v_template) : new Float32Array(this.vertexCount * 3);
        const faces = new Uint32Array(topology.faces);

        this.geometry = new THREE.BufferGeometry();
        this.positionAttribute = new THREE.BufferAttribute(positions, 3);
        this.geometry.setAttribute('position', this.positionAttribute);
        this.geometry.setIndex(new THREE.BufferAttribute(faces, 1));
        if (topology.skin_indices && topology.skin_weights) {
            this.geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(new Uint16Array(topology.skin_indices), 4));
            this.geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(new Float32Array(topology.skin_weights), 4));
        }
        this.geometry.computeVertexNormals();

        if (topology.joints && topology.parents && topology.skin_indices && topology.skin_weights) {
            this.restJoints = new Float32Array(topology.joints);
            this.mesh = new THREE.SkinnedMesh(this.geometry, this.material);
            this.initSkinnedSkeleton(this.restJoints, topology.parents);
        } else {
            this.mesh = new THREE.Mesh(this.geometry, this.material);
        }
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;
        this.mesh.frustumCulled = false;
        this.mesh.visible = false;
        this.scene.add(this.mesh);

        return topology;
    }

    initSkinnedSkeleton(restJoints, parents) {
        this.smplBones = [];
        for (let i = 0; i < parents.length; i++) {
            const bone = new THREE.Bone();
            bone.name = `smplx_${i}`;
            const parent = Number(parents[i]);
            const x = restJoints[i * 3];
            const y = restJoints[i * 3 + 1];
            const z = restJoints[i * 3 + 2];
            if (parent >= 0 && parent < i) {
                bone.position.set(
                    x - restJoints[parent * 3],
                    y - restJoints[parent * 3 + 1],
                    z - restJoints[parent * 3 + 2],
                );
                this.smplBones[parent].add(bone);
            } else {
                bone.position.set(x, y, z);
            }
            this.smplBones.push(bone);
        }
        if (this.smplBones[0]) this.mesh.add(this.smplBones[0]);
        this.smplSkeleton = new THREE.Skeleton(this.smplBones);
        this.mesh.bind(this.smplSkeleton);
    }

    updateMesh(flatVertices, root, flatJoints, smoothingAlpha = 1) {
        if (!this.mesh || !this.positionAttribute || !flatVertices) return;
        if (flatVertices.length !== this.vertexCount * 3) return;

        const positions = this.positionAttribute.array;
        const alpha = Math.max(0, Math.min(1, smoothingAlpha));
        if (!this.previousVertices || alpha >= 0.999) {
            positions.set(flatVertices);
        } else {
            for (let i = 0; i < positions.length; i++) {
                positions[i] = this.previousVertices[i] * (1 - alpha) + flatVertices[i] * alpha;
            }
        }
        this.previousVertices = new Float32Array(positions);
        this.positionAttribute.needsUpdate = true;
        this.geometry.computeVertexNormals();
        this.geometry.computeBoundingSphere();
        this.mesh.visible = true;

        if (flatJoints) this.updateSkeleton(flatJoints, alpha);
        if (root) this.updateTrail(root);
    }

    readFrame(packet, headerSize) {
        if (packet.length <= headerSize || !this.vertexCount) return null;
        const vertexValueCount = this.vertexCount * 3;
        const jointValueCount = 22 * 3;
        const paramsValueCount = 3 + 21 * 3 + 3 + jointValueCount;
        if (packet.length === headerSize + paramsValueCount) {
            let offset = headerSize;
            const rootOrient = packet.subarray(offset, offset + 3);
            offset += 3;
            const poseBody = packet.subarray(offset, offset + 21 * 3);
            offset += 21 * 3;
            const trans = packet.subarray(offset, offset + 3);
            offset += 3;
            return {
                frameId: Math.round(packet[0]),
                root: [packet[1], packet[2], packet[3]],
                bufferSize: Math.round(packet[7]),
                bufferCapacity: Math.round(packet[8]),
                rootOrient,
                poseBody,
                trans,
                joints: packet.subarray(offset, offset + jointValueCount),
            };
        }
        if (packet.length < headerSize + vertexValueCount + jointValueCount) return null;

        return {
            frameId: Math.round(packet[0]),
            root: [packet[1], packet[2], packet[3]],
            bufferSize: Math.round(packet[7]),
            bufferCapacity: Math.round(packet[8]),
            vertices: packet.subarray(headerSize, headerSize + vertexValueCount),
            joints: packet.subarray(headerSize + vertexValueCount, headerSize + vertexValueCount + jointValueCount),
        };
    }

    applyFrame(frame, smoothingAlpha = 1) {
        if (frame.rootOrient && frame.poseBody && frame.trans && this.smplBones.length) {
            this.updateSkinnedPose(frame.rootOrient, frame.poseBody, frame.trans, frame.root, frame.joints);
        } else {
            this.updateMesh(frame.vertices, frame.root, frame.joints, smoothingAlpha);
        }
    }

    updateSkinnedPose(rootOrient, poseBody, trans, root, flatJoints) {
        if (!this.mesh || !this.smplBones.length || !this.restJoints) return;
        this.setBoneAxisAngle(this.smplBones[0], rootOrient, 0);
        this.smplBones[0].position.set(
            this.restJoints[0] + trans[0],
            this.restJoints[1] + trans[1],
            this.restJoints[2] + trans[2],
        );
        for (let i = 1; i < this.smplBones.length; i++) {
            if (i <= 21) {
                this.setBoneAxisAngle(this.smplBones[i], poseBody, (i - 1) * 3);
            } else {
                this.smplBones[i].quaternion.identity();
            }
        }
        if (this.smplSkeleton) this.smplSkeleton.update();
        this.mesh.visible = true;
        if (flatJoints) this.updateSkeleton(flatJoints, 1);
        if (root) this.updateTrail(root);
    }

    setBoneAxisAngle(bone, values, offset) {
        const x = values[offset];
        const y = values[offset + 1];
        const z = values[offset + 2];
        const angle = Math.hypot(x, y, z);
        if (angle < 1e-8) {
            bone.quaternion.identity();
            return;
        }
        bone.quaternion.setFromAxisAngle(new THREE.Vector3(x / angle, y / angle, z / angle), angle);
    }

    initSkeletonOverlay() {
        const colors = [0xfeb21a, 0x00aaff, 0x134686, 0xffb600, 0x00d47e];
        this.jointMaterial = new THREE.MeshBasicMaterial({
            color: 0x00809d,
            depthTest: false,
            depthWrite: false,
        });
        this.boneMaterials = colors.map(color => new THREE.MeshBasicMaterial({
            color,
            depthTest: false,
            depthWrite: false,
        }));

        const jointGeometry = new THREE.SphereGeometry(0.026, 12, 12);
        for (let i = 0; i < 22; i++) {
            const joint = new THREE.Mesh(jointGeometry, this.jointMaterial);
            joint.renderOrder = 10;
            joint.visible = false;
            this.joints.push(joint);
            this.scene.add(joint);
        }

        for (let chainIndex = 0; chainIndex < this.chains.length; chainIndex++) {
            const material = this.boneMaterials[chainIndex];
            const chain = this.chains[chainIndex];
            for (let i = 0; i < chain.length - 1; i++) {
                this.boneConnections.push([chain[i], chain[i + 1]]);
                const bone = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.012, 0.012, 1, 8),
                    material,
                );
                bone.renderOrder = 10;
                bone.visible = false;
                this.bones.push(bone);
                this.scene.add(bone);
            }
        }
    }

    updateSkeleton(flatJoints, smoothingAlpha = 1) {
        if (flatJoints.length !== 22 * 3) return;
        const alpha = Math.max(0, Math.min(1, smoothingAlpha));
        const joints = new Float32Array(flatJoints.length);
        if (!this.previousJoints || alpha >= 0.999) {
            joints.set(flatJoints);
        } else {
            for (let i = 0; i < flatJoints.length; i++) {
                joints[i] = this.previousJoints[i] * (1 - alpha) + flatJoints[i] * alpha;
            }
        }
        this.previousJoints = new Float32Array(joints);

        for (let i = 0; i < 22; i++) {
            this.joints[i].position.set(joints[i * 3], joints[i * 3 + 1], joints[i * 3 + 2]);
            this.joints[i].visible = true;
        }

        for (let i = 0; i < this.boneConnections.length; i++) {
            const [startIndex, endIndex] = this.boneConnections[i];
            const start = new THREE.Vector3(
                joints[startIndex * 3],
                joints[startIndex * 3 + 1],
                joints[startIndex * 3 + 2],
            );
            const end = new THREE.Vector3(
                joints[endIndex * 3],
                joints[endIndex * 3 + 1],
                joints[endIndex * 3 + 2],
            );
            this.updateBone(this.bones[i], start, end);
            this.bones[i].visible = true;
        }
    }

    updateBone(bone, start, end) {
        const direction = new THREE.Vector3().subVectors(end, start);
        const length = direction.length();
        if (length < 0.001) return;
        bone.position.copy(new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5));
        bone.scale.y = length;
        bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
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
            opacity: 0.9,
        }));
        this.trail.frustumCulled = false;
        this.trail.visible = false;
        this.scene.add(this.trail);
    }

    updateTrail(root) {
        const next = { x: root[0], y: 0.018, z: root[2] };
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
            colors[i * 3] = 0.02 + 0.55 * alpha;
            colors[i * 3 + 1] = 0.55 + 0.35 * alpha;
            colors[i * 3 + 2] = 0.72 + 0.18 * alpha;
        }
        this.trailGeometry.attributes.position.needsUpdate = true;
        this.trailGeometry.attributes.color.needsUpdate = true;
        this.trailGeometry.setDrawRange(0, this.trailPoints.length);
        this.trail.visible = this.trailPoints.length > 1;
    }

    clearTrail() {
        this.trailPoints = [];
        this.previousVertices = null;
        this.previousJoints = null;
        this.trailGeometry.setDrawRange(0, 0);
        this.trail.visible = false;
        this.joints.forEach(joint => {
            joint.visible = false;
        });
        this.bones.forEach(bone => {
            bone.visible = false;
        });
    }

    setVisible(visible) {
        if (this.mesh) this.mesh.visible = visible;
        this.trail.visible = visible && this.trailPoints.length > 1;
        this.joints.forEach(joint => {
            joint.visible = visible;
        });
        this.bones.forEach(bone => {
            bone.visible = visible;
        });
    }
}

window.SMPLXAvatar = SMPLXAvatar;
