"use strict";
console.log("SCRIPT: Creating Neuron Generator");
console.log("=================================");

function NeuronGenerator() {
    this.neurons = [];
}

function LayerGenerator() {
    this.layer = [];
}

NeuronGenerator.prototype = {
    makeNeuron: function(numNeurons) {
        const neuron = document.createElement("div");
        neuron.className = "node";
        this.neurons.push(neuron);
        document.body.appendChild(neuron);
    },

}

LayerGenerator.prototype = {
    makeLayer: function(layerSize) {
        const neuronGenerator = new NeuronGenerator();
        for (let i = 0; i < layerSize; i++) {
            neuronGenerator.makeNeuron();
        }
        this.layer.append(neuronGenerator.neurons);
    }
}