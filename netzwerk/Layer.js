import { Matrix } from "./Matrix.js";
import { activations,loss } from "./functions.js";
import { predict, propagate, backward } from "./Propagations.js";

class Layer {
    /**
     * 
     * @param {String} type 
     * @param {Object} options 
     * @param {Layer} previousLayer 
     * @returns new Layer
     */
    constructor(type, options, previousLayer) {
        this.type = type;
        this.options = options;
        this.previous = previousLayer;
        this.channels = 1;

        switch(typeof this.options.actFunc) {
            case "string":
                this.actFunc = activations[this.options.actFunc];
                break;
            case "function":
                this.actFunc = this.options.actFunc;
                break;
            default:
                this.actFunc = (x) => x;
        }

        switch(typeof this.options.actFuncA) {
            case "string":
                this.actFuncA = activations[this.options.actFuncA];
                break;
            case "function":
                this.actFuncA = this.options.actFuncA;
                break;
            default:
                this.actFuncA = activations[this.options.actFunc + "A"] || function(x) { return 1 };
        }

        if(typeof this.actFuncA === undefined) throw new Error("Couldn't get derivative of activation function, specify yourself");

        this.forwardPass = predict[this.type].bind(this);
        this.forwardProp = propagate[this.type].bind(this);
        this.backProp = backward[this.type].bind(this);

        if(this.type === "conv") {

            this.options.padding = options.padding || 0;
            this.options.stride = options.stride || 1;

            this.filter = [];
            this.biases = [];
            this.channels = this.options.channels;
            this.iDim = (previousLayer.iDim + 2*this.options.padding - this.options.size) / this.options.stride + 1;
            this.jDim = (previousLayer.jDim + 2*this.options.padding - this.options.size) / this.options.stride + 1;

            for(let i = 0;i < this.options.channels;i++) {
                this.filter.push([]);
                this.biases.push([]);
                for(let j = 0;j < this.previous.channels;j++) {
                    this.filter[i].push(new Matrix(this.options.size, this.options.size));
                    this.filter[i][j].randomize();
                    this.biases[i].push(Math.random()*2-1);
                }
            }
            
        }else if(this.type === "dens") {
            this.weight = new Matrix(this.previous.jDim, this.options.size);
            this.weight.randomize();
            this.bias = new Matrix(1, this.options.size);
            this.bias.randomize();

            this.iDim = 1;
            this.jDim = this.options.size;
        }else if(this.type === "flatten") {
            this.iDim = 1;
            this.jDim = this.previous.iDim*this.previous.jDim*this.previous.channels;
            this.channels = 1;
        }else if(this.type === "maxpool") {
            this.iDim = (previousLayer.iDim + 2*options.padding - options.size) / options.stride + 1;
            this.jDim = (previousLayer.jDim + 2*options.padding - options.size) / options.stride + 1;

            if(!Number.isInteger(this.iDim)) {
                console.warn("Rounding down maxpool size");
                this.iDim = Math.floor(this.iDim);
                this.jDim = Math.floor(this.jDim);
            }

            this.channels = this.previous.channels;
        }else if(this.type === "dropout") {
            this.iDim = this.previous.iDim;
            this.jDim = this.previous.jDim;
            if(this.options.rate > 1 || this.options.rate < 0) {
                throw new Error("Dropout rate must be between 0 and 1");
            }
        }
        
    }
}

export { Layer }