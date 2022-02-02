import { Matrix } from "./Matrix.js";
import { activations,loss } from "./functions.js";
import { Layer } from "./Layer.js";

class Netzwerk {
    /**
     * Creates 
     * @param {Number/Array/Object} inputSize 
     */
    constructor(inputSize) {
        if(inputSize?.json === true) {
            this.fromJSON(inputSize);
            return;
        }
        if(typeof inputSize === "number") {
            this.iDim = 1;
            this.jDim = inputSize;
            this.channels = 0;
        }else if(typeof inputSize.iDim !== "undefined") {
            this.iDim = inputSize.iDim;
            this.jDim = inputSize.jDim;
            this.channels = inputSize.channels || 1;
        }else {
            this.iDim = inputSize[0];
            this.jDim = inputSize[1];
            this.channels = inputSize[2] || 1;
        }
        this.layers = [];
        this.layerCount = 0;
    }

    /**
     * Adds Layer to Network 
     * @param {String} type 
     * @param {Object} options 
     */
    addLayer(type, options) {
        let prev = this.layers[this.layers.length-1];
        if(!this.layerCount) {
            prev = {
                "iDim": this.iDim,
                "jDim": this.jDim,
                "channels": this.channels
            }
            this.layerCount = 0;
        }
        this.layers.push(new Layer(type, options, prev));
        this.layerCount++;
    }

    /**
     * Handles input formating
     * @param {Array} input 
     * @param {number} batch size 
     * @returns {Array | Matrix} input for Network
     */
    handleInput(input, batch = 1) {
        let erg;

        if(this.layers[0].type === "dens") {
            if(input.werte) {
                erg = input;
            }else if(typeof input[0] === "number") {
                erg = new Matrix(1, this.layers[0].previous.jDim);
                erg.werte = input;
            }else if(typeof input[0][0] === "number") {
                erg = new Matrix(batch, this.layers[0].previous.jDim);
                erg.werte = [];
                for(let i = 0;i < batch;i++) {
                    erg.werte.push(...input(i));
                }
            }else {
                throw new Error("Invalid input format");
            }
        }else {
            if(input[0]?.werte) {
                erg = input;
            }else if(input?.werte) {
                erg = [input];
            }else if(input?.length === this.layers[0].channels) {
                throw new Error("input not supported");
            }else {
                throw new Error("input not supported");
            }
        }

        return erg;
    }
    
    /**
     * Evaluates input
     * @param {Matrix/Array} input 
     * @returns {Array} output
     */
    predict(input) {
        let erg = this.handleInput(input);
        let self = this;
        for(let layerIndex = 0;layerIndex < this.layers.length;layerIndex++) {
            erg = this.layers[layerIndex].forwardPass(erg);
        }

        return erg;
    }

    /**
     * Sets error function e.g crossEntropy
     * @param {string/function} func 
     */
    setErrorFunction(func) {
        switch(typeof func) {
            case "string":
                this.eTotalString = func;
                this.eTotal = loss[func];
                break;
            case "function":
                this.eTotal = func;
            default:
                throw new Error("Can't handle function type of " + (typeof func));
        }
    }

    /**
     * Derivative of error function
     * @param {string/functiom} func 
     */
    setLossFunction(func) {
        switch(typeof func) {
            case "string":
                this.lossString = func;
                this.loss = loss[func];
                break;
            case "function":
                this.loss = func;
            default:
                throw new Error("Can't handle function type of " + (typeof func));
        }
    }

    /**
     * Trains network
     * @param {Matrix/Array} input 
     * @param {Matrix} target 
     * @param {Object} opts 
     * @returns Error
     */
    train(input, target, opts) {
        let output = input;

        /*TODO: Handle input */
        if(this.layers[0].type !== "dens") output = [output];

        this.layers[0].previous.out = output;
        for(let i = 0;i < this.layers.length;i++) {
            output = this.layers[i].forwardProp(output, opts.batch);
        }
        let Error = this.eTotal(output, target);
        let loss = this.loss(output, target);

        let gradient = loss;
        for(let i = 0; i < this.layers.length;i++) {
            gradient = this.layers[this.layers.length-1-i].backProp(gradient, opts);
        }

        return Error;
    }
    
    /**
     * Trains network on set
     * @param {Array} input 
     * @param {Array} target 
     * @param {Object} opts 
     * @param {number} rounds 
     * @returns error
     */
    trainSet(input, target, opts, rounds = 100) {
        let roundIndex = 0;
        let error = 0;
        while(roundIndex < rounds) {
            roundIndex++;
            let eSum = 0;
            //console.log("Starte Runde", roundIndex);
            for(let i = 0;i < input.length;i++) {
                let e = this.train(input[i], target[i], opts);
                eSum += e;
                //console.log("Durch", i, "inputs von", input.length);
                //console.log("Fehler:", e)
            }
            eSum /= input.length;
            //console.log("Runde", roundIndex, "finished");
            //console.log("Error: ", eSum);
            error += eSum;
        }

        return error/roundIndex;
    } 

    toJSON() {
        let erg = {
            "json": true,
            "iDim": this.iDim,
            "jDim": this.jDim,
            "channels": this.channels,
            "eTotal": this.eTotalString,
            "loss": this.lossString,
            "layers": []
        }
        for(let i = 0;i < this.layers.length;i++) {
            let layer = this.layers[i];
            let schicht = {
                "type": layer.type,
                "options": layer.options,
                "previous": layer.previous
            }
            if(layer.type === "dens") {
                schicht.weight = layer.weight.toJSON();
                schicht.bias = layer.bias.toJSON();
            }else if(layer.type === "conv") {
                schicht.filter = layer.filter;
                schicht.biases = layer.biases;
            }
            erg.layers.push(schicht);
        }
        return erg;
    }

    fromJSON(json) {
        this.iDim = json.iDim;
        this.jDim = json.jDim;
        this.channels = json.channels;
        this.setErrorFunction(json.eTotal);
        this.setLossFunction(json.loss);

        this.layers = [];
        for(let i = 0;i < json.layers.length;i++) {
            let layer = json.layers[i];
            let prev = i === 0 ? layer.previous : this.layers[i-1];
            this.addLayer(layer.type, layer.options, prev);
            if(layer.type === "dens") {
                let w = new Matrix(1,1);
                w.fromJSON(layer.weight);
                this.layers[i].weight = w;
                let b = new Matrix(1,1);
                b.fromJSON(layer.bias);
                this.layers[i].bias = b;
            }else if(layer.type === "conv") {
                this.layers[i].filter = layer.filter;
                this.layers[i].biases = layer.biases;
            }
        }
    
    }

}



export { Netzwerk }