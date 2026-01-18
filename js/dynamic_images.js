import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "sg.TextEncodeReferenceImages.DynamicImages",
    async nodeCreated(node) {
        if (node.comfyClass === "TextEncodeReferenceImages") {
            const onConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, link_info) {
                onConnectionsChange?.apply(this, arguments);

                if (type !== 1) return; // Only care about input connections

                // 1. Find all image inputs
                const getImageInputs = () => this.inputs.filter(i => i.name.startsWith("image"));
                let imageInputs = getImageInputs();

                // 2. If last image input is connected, add a new one
                if (imageInputs.length > 0) {
                    const lastImageInput = imageInputs[imageInputs.length - 1];
                    if (lastImageInput.link !== null) {
                        const lastNum = parseInt(lastImageInput.name.match(/\d+/)[0]);
                        this.addInput(`image${lastNum + 1}`, "IMAGE", { optional: true });
                    }
                } else {
                    this.addInput("image1", "IMAGE", { optional: true });
                }

                // 3. Remove trailing empty image inputs, keeping exactly one
                imageInputs = getImageInputs();
                let lastIdx = imageInputs.length - 1;
                while (lastIdx > 0) {
                    const current = imageInputs[lastIdx];
                    const prev = imageInputs[lastIdx - 1];

                    if (current.link === null && prev.link === null) {
                        this.removeInput(this.inputs.indexOf(current));
                        imageInputs = getImageInputs();
                        lastIdx--;
                    } else {
                        break;
                    }
                }
            };
        }
    },
});
