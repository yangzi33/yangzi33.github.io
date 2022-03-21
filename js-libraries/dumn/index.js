"use strict";
function generate() {
    const numNeurons = document.getElementById("numNeurons").value;
    const lg = new LayerGenerator();
    lg.makeLayer(numNeurons);
}