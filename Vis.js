import { Matrix } from "./Matrix.js";
import { Netzwerk } from "./Netzwerk.js";

class Visualisierer {
    /**
     * Create new visulaiser for a network
     * @param {Netzwerk} netz 
     */
    constructor(netz) { 
        this.network = netz;
        this.errors = [];
    }

    /**
     * Set targets to be rendered when Network is mapped to a canvas
     * @param {Array} inputs 
     * @param {Array} outputs 
     */
    setTargets(inputs, outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    /**
     * Renders set targets to canvas (automaticly done by mapToCanvas when specified)
     * @param {String} canvasId 
     */
    renderTargets(canvasId) {
        let canvas = document.getElementById(canvasId);
        let ctx = canvas.getContext("2d");

        for(let i = 0;i < this.inputs.length;i++) {
            let x = (this.inputs[i][0]+0.5)*canvas.width;
            let y = (this.inputs[i][1]+0.5)*canvas.height;
            ctx.beginPath();
            ctx.strokeStyle = this.outputs[i][0] > 0.5 ? "#FF6666" : "#6666FF";
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = ctx.strokeStyle;
            ctx.stroke();
        }
    }

    /**
     * Maps Network predictions to canvas and renders targets when specified
     * @param {String} canvasId 
     * @param {Boolean} targets 
     */
    mapToCanvas(canvasId, targets = false) {
        let canvas = document.getElementById(canvasId);
        canvas.setAttribute('width', window.getComputedStyle(canvas, null).getPropertyValue("width"));
        canvas.setAttribute('height', window.getComputedStyle(canvas, null).getPropertyValue("height"));

        let ctx = canvas.getContext("2d");

        let matrix = new Matrix(1, this.network.inputCount);

        for(let x = 0;x < canvas.width;x += 2) {
            for(let y = 0;y < canvas.height;y += 2) {
                let g = 0;
                matrix.werte = [(x/canvas.width)-0.5,(y/canvas.height)-0.5];
                let erg = this.network.predict(matrix).werte[0];
                   
                let r = erg > 0.5 ? Math.min(erg*100,255) : 0;
                let b = erg <= 0.5 ? Math.min(erg*100, 255) : 0;                
                
                let a = 255;
                ctx.fillStyle = "rgba("+r+","+g+","+b+","+(a/255)+")";
                ctx.fillRect(x,y,2,2);
            }
        }
        if(targets && this.inputs.length) {
            this.renderTargets(canvasId);
        }
    }

    /**
     * Renders Networklayout to Canvas
     * @param {String} canvasId 
     */
    drawToCanvas(canvasId) {
        let canv = document.getElementById(canvasId);
        canv.setAttribute('width', window.getComputedStyle(canv, null).getPropertyValue("width"));
        canv.setAttribute('height', window.getComputedStyle(canv, null).getPropertyValue("height"));
        let ctx = canv.getContext("2d");

        let maxNeuronCount = this.network.weights[0].iDim;
        for(let i = 0;i < this.network.weights.length;i++) {
            maxNeuronCount = Math.max(maxNeuronCount, this.network.weights[i].jDim);
        }

        let xScale = canv.width/this.network.weights.length;
        let yScale = canv.height/maxNeuronCount;

        let radiusRespectToHeight = yScale/2;
        let radiusRespectToWidth = xScale/2;
        let neuronRadius = Math.min(radiusRespectToHeight, radiusRespectToWidth);

        ctx.font = "6px Arial";
        ctx.textAlign = "center";

        for(let weightIndex = 0;weightIndex < this.network.weights.length;weightIndex++) {
            for(let iIndex = 0;iIndex < this.network.weights[weightIndex].iDim;iIndex++) {
                //Weights from i to j
                for(let jIndex = 0;jIndex < this.network.weights[weightIndex].jDim;jIndex++) {
                    ctx.beginPath();
                    ctx.moveTo(xScale*weightIndex, yScale*iIndex + neuronRadius);
                    ctx.lineTo(xScale*(weightIndex+1), yScale*jIndex + neuronRadius);
                    ctx.stroke();
                }
                //Neurones i
                ctx.beginPath();
                ctx.arc(xScale*weightIndex, yScale*iIndex + neuronRadius, neuronRadius, 0, 2 * Math.PI);
                ctx.fillStyle = "#888888";
                ctx.fill();
                ctx.stroke();
            }
        }

        for(let jIndex = 0;jIndex < this.network.weights[this.network.weights.length-1].jDim;jIndex++) {
            ctx.beginPath();
            ctx.arc(xScale*this.network.weights.length, yScale*jIndex + neuronRadius, neuronRadius, 0, 2 * Math.PI);
            ctx.fillStyle = "#888888";
            ctx.fill();
            ctx.stroke();
        }
    }

    /**
     * Plots all previous errors + new error to canvas that fit 
     * @param {String} canvasId 
     * @param {Number} nError 
     */
    plotError(canvasId, nError) {
        let canvas = document.getElementById(canvasId);
        canvas.setAttribute('width', window.getComputedStyle(canvas, null).getPropertyValue("width"));
        canvas.setAttribute('height', window.getComputedStyle(canvas, null).getPropertyValue("height"));
        let ctx = canvas.getContext("2d");
        let height = canvas.height;
        let width = canvas.width;
        
        this.errors.push(nError);
        if(this.errors.length > width) this.errors.shift();

        ctx.strokeStyle = "black";

        let maxError = this.errors.reduce((a,b)=>{
            return Math.max(a,b);
        });
        let scale = height/maxError;

        ctx.beginPath();
        ctx.moveTo(0,height-scale*this.errors[0]);
        for(let x = 1;x < width-1;x++) {            
            ctx.lineTo(x,height-scale*this.errors[x+1])
        }
        ctx.stroke();
    }

    trainSetShowCMD(input, target, opts, rounds) {
        process.stdout.write('\x1B[?25l');
        let roundIndex = 0;
        let error = 0;
        while(roundIndex < rounds) {
            roundIndex++;
            let eSum = 0;
            for(let i = 0;i < input.length;i++) {
                let e = this.network.train(input[i], target[i], opts);
                eSum += e;
            }
            eSum /= input.length;
            error += eSum;
            this.writeProgressLine(roundIndex, rounds);
        }
        process.stdout.write('\x1B[?25h');
        return error/roundIndex;
    }

    sleep(time) { return new Promise((r) => setTimeout(r,time)) }
    /**
     * Clears line and writes message
     * @param {number} curr Value
     * @param {number} max Value
     */
    writeProgressLine(curr, max) {
        
        let scalar = 20;

        let oCurr = curr;
        let oMax = max;

        if(max !== scalar) {
            let scale = scalar/max;
            max *= scale;
            curr *= scale;
            max = Math.floor(max);
            curr = Math.floor(curr);
        }

        function resetLine() {
            //process.stdout.clearLine();
            process.stdout.cursorTo(0); 
        }
        
        let string = "";
        for(let i = 0;i < max;i++) {
            if(i < curr) {
                string += "=";
            }else {
                string += ".";
            }
        }
        string += ">| ";
        string += oCurr + " from " + oMax;

        resetLine();

        process.stdout.write(string);
    }

    
}

export { Visualisierer };
