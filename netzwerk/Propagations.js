import { Matrix } from "./Matrix.js";
/**global
 * TODO:
 * Vektorfizierung
 */

let convForward = function(input) {
    let erg = [];

    for(let i = 0;i < this.filter.length;i++) {
        erg.push(new Matrix(this.iDim, this.jDim));
        erg[i].init();
        for(let j = 0;j < this.filter[i].length;j++) {
            let zwischenErg = input[j].conv(this.filter[i][j], this.options.stride, this.options.padding ); //might be input[i] instead but unlickely
            zwischenErg = zwischenErg.add(this.biases[i][j]);
            zwischenErg = this.actFunc(zwischenErg);
            erg[i] = erg[i].add(zwischenErg);
        }
        
    }

    return erg;
}

let convPropagate = function(input) {
    let erg = [];
    this.net = [];

    for(let i = 0;i < this.filter.length;i++) {
        erg.push(new Matrix(this.iDim, this.jDim));
        erg[i].init();

        this.net.push(new Matrix(this.iDim, this.jDim));
        this.net[i].init();

        for(let j = 0;j < this.filter[i].length;j++) {
            let zwischenErg = input[j].conv(this.filter[i][j], this.options.stride, this.options.padding ); //might be input[i] instead but unlickely
            zwischenErg = zwischenErg.add(this.biases[i][j]);

            this.net[i] = this.net[i].add(zwischenErg);
            
            zwischenErg = this.actFunc(zwischenErg);
            erg[i] = erg[i].add(zwischenErg);
        }
        
    }

    this.out = erg;
    return erg;
}

//Best source: http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/

//TODO: clump wDeltas /later batch training

let convBackward = function(delta, opts) {
    let currDeltas = [];
    for(let i = 0;i < delta.length;i++) {
        currDeltas.push(this.actFuncA(this.net[i]).dot(delta[i]));
    }  

    let wGrads = [];
    let bGrads = [];
    let newDeltas = [];

    for(let i = 0;i < this.previous.channels;i++) {
        newDeltas.push(new Matrix(this.previous.iDim, this.previous.jDim));
        newDeltas[i].init();
    }

    for(let i = 0;i < this.previous.out.length;i++) {
        wGrads.push([]);
        bGrads.push([]);
        
        for(let j = 0;j < currDeltas.length;j++) {
            let grad = this.previous.out[i].conv(currDeltas[j], this.options.stride, this.options.padding);
            wGrads[i].push(grad.mult(opts.lernRate));
            bGrads[i].push(currDeltas[j].sum() * opts.lernRate);

            let upscaleDelta = currDeltas[j].upscale(this.filter[j][i], this.options.stride, this.options.padding);
            newDeltas[i] = newDeltas[i].add(upscaleDelta);
            
            this.filter[j][i] = this.filter[j][i].sub(wGrads[i][j]);
            this.biases[j][i] = this.biases[j][i] - bGrads[i][j];
        }
    }
    return newDeltas;
}

/* ------------------------------------------------------------------------------------------------------------------ */

let maxpoolForward = function(input) {

    let erg = [];

    for(let inputIndex = 0;inputIndex < input.length;inputIndex++) {
        erg.push(new Matrix(this.iDim, this.jDim));

        for(let ergI = 0;ergI < this.iDim;ergI++) {
            for(let ergJ = 0;ergJ < this.jDim;ergJ++) {
                let max = 0;
                for(let i = 0 - this.options.padding;i < this.options.size + this.options.padding;i++) {
                    for(let j = 0 - this.options.padding;j < this.options.size + this.options.padding;j++) {
                        let currentValue = input[inputIndex].at(i + ergI * this.options.stride ,j + ergJ  * this.options.stride , true);
                        if(currentValue > max) max = currentValue;
                    }
                }

                erg[inputIndex].set(ergI, ergJ, max);
            }
        }

    }

    return erg;
}

let maxpoolPropagate = function(input) {
    let erg = [];
    this.mask = [];
    for(let inputIndex = 0;inputIndex < input.length;inputIndex++) {
        erg.push(new Matrix(this.iDim, this.jDim));
        //Loop over output
        for(let ergI = 0;ergI < this.iDim;ergI++) {
            for(let ergJ = 0;ergJ < this.jDim;ergJ++) {
                let max = 0;
                let maxIndex = [0,0];
                //Selecting Max convolution
                for(let i = 0 - this.options.padding;i < this.options.size + this.options.padding;i++) {
                    for(let j = 0 - this.options.padding;j < this.options.size + this.options.padding;j++) {
                        let currentValue = input[inputIndex].at(i + ergI * this.options.stride ,j + ergJ  * this.options.stride , true);
                        if(currentValue > max) {
                            max = currentValue;
                            maxIndex = [i,j];
                        }
                    }
                }
                erg[inputIndex].set(ergI, ergJ, max);
                this.mask.push(maxIndex);
            }
        }

    }
    this.out = erg;
    return erg;
}

let maxpoolBackward = function(delta, opts) {
    let newDeltas = [];
    let filter = new Matrix(this.options.size, this.options.size);
    for(let i = 0;i < delta.length;i++) {
        filter.init();
        filter.set(...this.mask[i], 1);
        let upscaleDelta = delta[i].upscale(filter, this.options.stride, this.options.padding);
        newDeltas.push(upscaleDelta);
    }
    return newDeltas;
}

/* ------------------------------------------------------------------------------------------------------------------ */

let densForward = function(input) {
    let net = input.mult(this.weight);
    net = net.add(this.bias);
    let out = this.actFunc(net);
    return out;
}

let densPropagate = function(input) {
    this.net = input.mult(this.weight);
    let b = new Matrix(input.iDim, this.options.size);
    b.werte = [];
    for(let i = 0;i < input.iDim;i++) {
        b.werte.push(...this.bias.werte);
    }
    this.net = this.net.add(b);
    this.out = this.actFunc(this.net);
    return this.out;
}

let densBackward = function(delta, opts) {
    let newDelta = delta.dot(this.actFuncA(this.net));
    let transOut = this.previous.out.transpose();
    let adjust = transOut.mult(newDelta);
    this.dWeight = adjust.mult(opts.lernRate);
    this.dBias = newDelta.mult(opts.lernRate);

    newDelta = newDelta.mult(this.weight.transpose());
    this.dBias = this.dBias.collapse();

    /*console.log(this.weight.toString());
    console.log(this.weight.iDim, this.weight.jDim);
    console.log(this.bias.toString());
    console.log("");
    if(this.dWeight.sum((x)=>Math.abs(x)) === 0) {
        throw "force exit";
    }*/

    this.weight = this.weight.sub(this.dWeight);
    this.bias = this.bias.sub(this.dBias);


    return newDelta;
}

/* ------------------------------------------------------------------------------------------------------------------ */

let flattenForward = function(input) {
    let erg = new Matrix(this.iDim, this.jDim);
    erg.werte = [];

    for(let i = 0;i < input.length;i++) {
        erg.werte.push(...input[i].werte);
    }

    return erg;
}

let flattenPropagate = function(input) {
    let erg = new Matrix(this.iDim, this.jDim);
    erg.werte = [];

    for(let i = 0;i < input.length;i++) {
        erg.werte.push(...input[i].werte);
    }
    this.out = erg;
    return erg;
}

let flattenBackward = function(delta, opts) {
    let newDeltas = [];
    let bundleSize = this.previous.iDim * this.previous.jDim;
    for(let i = 0;i < this.previous.channels;i++) {
        newDeltas.push(new Matrix(this.previous.iDim, this.previous.jDim));
        newDeltas[i].werte = delta.werte.slice(i*bundleSize, (i+1) * bundleSize);
    }
    return newDeltas;
}

/* ------------------------------------------------------------------------------------------------------------------ */

let dropoutForward = function(input) {
    return input;
}

let dropoutPropagate = function(input) {
    let erg = new Matrix(input.iDim, input.jDim);
    this.drop = new Matrix(input.iDim, input.jDim);
    this.drop.werte = input.werte.map((e)=>{
        return Math.random() > 1-this.options.rate ? 1 : 0
    });
    erg = input.dot(this.drop);
    erg.div(this.options.rate); //scaling a.k.a inverted dropout (instead of only dropout)
    this.out = erg;
    return erg;
}

let dropoutBackward = function(delta, opts) {
    return delta.dot(this.drop);
}

/* ------------------------------------------------------------------------------------------------------------------ */

let predict = {
    "conv": convForward,
    "dens": densForward,
    "flatten": flattenForward,
    "maxpool": maxpoolForward,
    "dropout": dropoutForward
}

let propagate = {
    "conv": convPropagate,
    "dens": densPropagate,
    "flatten": flattenPropagate,
    "maxpool": maxpoolPropagate,
    "dropout": dropoutPropagate
}

let backward = {
    "conv": convBackward,
    "dens": densBackward,
    "flatten": flattenBackward,
    "maxpool": maxpoolBackward,
    "dropout": dropoutBackward
}


export {
    predict,
    propagate,
    backward
}