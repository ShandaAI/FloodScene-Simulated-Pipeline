class BaseAvatar {
    constructor(scene) {
        this.scene = scene;
        this.displayName = 'Renderer';
    }

    async loadTopology() {
        throw new Error('loadTopology() must be implemented by avatar subclasses');
    }

    readFrame() {
        throw new Error('readFrame() must be implemented by avatar subclasses');
    }

    applyFrame() {
        throw new Error('applyFrame() must be implemented by avatar subclasses');
    }

    clearTrail() {}

    setVisible() {}
}

window.BaseAvatar = BaseAvatar;

