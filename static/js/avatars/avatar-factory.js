class AvatarFactory {
    static fromConfig(config, scene) {
        return AvatarFactory.create(config?.renderer || config?.visualization || 'g1', scene);
    }

    static create(rendererName, scene) {
        const normalized = String(rendererName || 'g1').toLowerCase();
        if (normalized.includes('smplh') || normalized.includes('flooddiffusion')) return new SMPLHAvatar(scene);
        if (normalized.includes('smplx')) return new SMPLXAvatar(scene);
        if (normalized.includes('g1') || normalized.includes('unitree')) return new G1Avatar(scene);
        throw new Error(`Unknown renderer: ${rendererName}`);
    }
}

window.AvatarFactory = AvatarFactory;
