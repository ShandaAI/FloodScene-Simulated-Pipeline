class G1Avatar extends BaseAvatar {
    constructor(scene) {
        super(scene);
        this.scene = scene;
        this.displayName = 'Unitree G1';
        this.jointCount = 0;
        this.meshItems = [];
        this.meshes = [];
        this.previousJoints = null;
        this.trailPoints = [];
        this.maxTrailPoints = 240;
        this.showMesh = true;
        this.showSkeletonOverlay = false;
        this.showTrail = true;
        this.joints = [];
        this.bones = [];
        this.boneConnections = [];
        this.chains = [];

        this.bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0xb8c0c7,
            roughness: 0.48,
            metalness: 0.28,
            side: THREE.DoubleSide,
        });
        this.darkMaterial = new THREE.MeshStandardMaterial({
            color: 0x20262b,
            roughness: 0.58,
            metalness: 0.18,
            side: THREE.DoubleSide,
        });
        this.accentMaterial = new THREE.MeshStandardMaterial({
            color: 0x52616b,
            roughness: 0.5,
            metalness: 0.2,
            side: THREE.DoubleSide,
        });

        this.initTrail();
    }

    async loadTopology() {
        const response = await fetch('/api/g1/topology');
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Unitree G1 topology failed to load');
        }
        if (!THREE.STLLoader) {
            throw new Error('THREE.STLLoader is not available');
        }

        const topology = await response.json();
        this.jointCount = topology.joint_count;
        this.chains = topology.chains || [];
        this.initSkeletonOverlay();

        if (!this.showMesh) return topology;

        const loader = new THREE.STLLoader();
        const meshItems = topology.mesh_items || [];
        this.meshItems = meshItems.map(item => ({
            jointIdx: item.joint_idx,
            geomPosition: item.geom_pos,
            geomRotation: item.geom_rot,
            meshFile: item.mesh_file,
            meshUrl: item.mesh_url,
        }));

        await Promise.all(this.meshItems.map(item => this.loadMeshItem(loader, item)));
        return topology;
    }

    loadMeshItem(loader, item) {
        return new Promise((resolve, reject) => {
            loader.load(
                item.meshUrl,
                geometry => {
                    this.convertMujocoGeometryToKimodo(geometry);
                    geometry.computeVertexNormals();
                    geometry.computeBoundingSphere();

                    const mesh = new THREE.Mesh(geometry, this.materialForMesh(item.meshFile));
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    mesh.frustumCulled = false;
                    mesh.matrixAutoUpdate = false;
                    mesh.visible = false;
                    item.mesh = mesh;
                    this.meshes.push(mesh);
                    this.scene.add(mesh);
                    resolve(mesh);
                },
                undefined,
                error => reject(error),
            );
        });
    }

    materialForMesh(meshFile) {
        if (meshFile.includes('rubber') || meshFile.includes('pelvis')) return this.darkMaterial;
        if (meshFile.includes('logo') || meshFile.includes('head')) return this.accentMaterial;
        return this.bodyMaterial;
    }

    convertMujocoGeometryToKimodo(geometry) {
        const positions = geometry.getAttribute('position');
        const array = positions.array;
        for (let i = 0; i < array.length; i += 3) {
            const x = array[i];
            const y = array[i + 1];
            const z = array[i + 2];
            array[i] = y;
            array[i + 1] = z;
            array[i + 2] = x;
        }
        positions.needsUpdate = true;
    }

    updatePose(flatJoints, flatRotations, root, smoothingAlpha = 1) {
        if (!flatJoints || !flatRotations) return;
        if (flatJoints.length !== this.jointCount * 3) return;
        if (flatRotations.length !== this.jointCount * 9) return;

        const joints = this.smoothJoints(flatJoints, smoothingAlpha);
        if (this.showMesh) {
            for (const item of this.meshItems) {
                if (!item.mesh) continue;
                const jointIdx = item.jointIdx;
                const jointPos = [
                    joints[jointIdx * 3],
                    joints[jointIdx * 3 + 1],
                    joints[jointIdx * 3 + 2],
                ];
                const jointRot = this.readMatrix3(flatRotations, jointIdx);
                const meshRot = this.multiplyMat3(jointRot, item.geomRotation);
                const meshOffset = this.applyMat3(jointRot, item.geomPosition);
                const meshPos = [
                    jointPos[0] + meshOffset[0],
                    jointPos[1] + meshOffset[1],
                    jointPos[2] + meshOffset[2],
                ];
                this.applyMeshMatrix(item.mesh, meshRot, meshPos);
                item.mesh.visible = true;
            }
        }

        if (this.showSkeletonOverlay) this.updateSkeleton(joints);
        if (root && this.showTrail) this.updateTrail(root);
    }

    readFrame(packet, headerSize) {
        if (packet.length <= headerSize || !this.jointCount) return null;
        const jointValueCount = this.jointCount * 3;
        const rotationValueCount = this.jointCount * 9;
        if (packet.length < headerSize + jointValueCount + rotationValueCount) return null;

        return {
            frameId: Math.round(packet[0]),
            root: [packet[1], packet[2], packet[3]],
            bufferSize: Math.round(packet[7]),
            bufferCapacity: Math.round(packet[8]),
            joints: packet.subarray(headerSize, headerSize + jointValueCount),
            rotations: packet.subarray(headerSize + jointValueCount, headerSize + jointValueCount + rotationValueCount),
        };
    }

    applyFrame(frame, smoothingAlpha = 1) {
        this.updatePose(frame.joints, frame.rotations, frame.root, smoothingAlpha);
    }

    smoothJoints(flatJoints, smoothingAlpha) {
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
        return joints;
    }

    readMatrix3(flatRotations, jointIdx) {
        const offset = jointIdx * 9;
        return [
            flatRotations[offset],
            flatRotations[offset + 1],
            flatRotations[offset + 2],
            flatRotations[offset + 3],
            flatRotations[offset + 4],
            flatRotations[offset + 5],
            flatRotations[offset + 6],
            flatRotations[offset + 7],
            flatRotations[offset + 8],
        ];
    }

    multiplyMat3(a, b) {
        return [
            a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
            a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
            a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
            a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
            a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
            a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
            a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
            a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
            a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
        ];
    }

    applyMat3(m, v) {
        return [
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        ];
    }

    applyMeshMatrix(mesh, rotation, position) {
        mesh.matrix.set(
            rotation[0], rotation[1], rotation[2], position[0],
            rotation[3], rotation[4], rotation[5], position[1],
            rotation[6], rotation[7], rotation[8], position[2],
            0, 0, 0, 1,
        );
        mesh.matrixWorldNeedsUpdate = true;
    }

    initSkeletonOverlay() {
        const colors = [0x00aaff, 0xfeb21a, 0x134686, 0x00d47e, 0xffb600];
        this.jointMaterial = new THREE.MeshBasicMaterial({
            color: 0xd62828,
            transparent: true,
            opacity: 0.9,
            depthTest: true,
            depthWrite: false,
        });
        this.boneMaterials = colors.map(color => new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.85,
            depthTest: true,
            depthWrite: false,
        }));

        const jointGeometry = new THREE.SphereGeometry(0.018, 10, 10);
        for (let i = 0; i < this.jointCount; i++) {
            const joint = new THREE.Mesh(jointGeometry, this.jointMaterial);
            joint.renderOrder = 10;
            joint.visible = false;
            this.joints.push(joint);
            this.scene.add(joint);
        }

        for (let chainIndex = 0; chainIndex < this.chains.length; chainIndex++) {
            const chain = this.chains[chainIndex];
            const material = this.boneMaterials[chainIndex % this.boneMaterials.length];
            for (let i = 0; i < chain.length - 1; i++) {
                this.boneConnections.push([chain[i], chain[i + 1]]);
                const bone = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.008, 0.008, 1, 8),
                    material,
                );
                bone.renderOrder = 10;
                bone.visible = false;
                this.bones.push(bone);
                this.scene.add(bone);
            }
        }
    }

    updateSkeleton(joints) {
        for (let i = 0; i < this.jointCount; i++) {
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
        if (this.trailPoints.length > this.maxTrailPoints) this.trailPoints.shift();

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
        this.previousJoints = null;
        this.trailGeometry.setDrawRange(0, 0);
        this.trail.visible = false;
        this.meshes.forEach(mesh => {
            mesh.visible = false;
        });
        this.joints.forEach(joint => {
            joint.visible = false;
        });
        this.bones.forEach(bone => {
            bone.visible = false;
        });
    }

    setVisible(visible) {
        this.meshes.forEach(mesh => {
            mesh.visible = visible && this.showMesh;
        });
        this.trail.visible = visible && this.showTrail && this.trailPoints.length > 1;
        this.joints.forEach(joint => {
            joint.visible = visible && this.showSkeletonOverlay;
        });
        this.bones.forEach(bone => {
            bone.visible = visible && this.showSkeletonOverlay;
        });
    }
}

window.G1Avatar = G1Avatar;
