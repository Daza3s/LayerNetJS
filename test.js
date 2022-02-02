import { Matrix } from "./netzwerk/Matrix.js";
import { Netzwerk } from "./netzwerk/Netzwerk.js"
import { predict } from "./netzwerk/Propagations.js";
import mnist from "mnist";
import { activations } from "./netzwerk/functions.js";
import { in1, in2, in3, in4, tar1, tar2, tar3, tar4, batchIn, batchTar } from "./netzwerk/xorMatrixen.js";
import * as fs from "fs";

/**
 * TODO: Fix batch, implement dropout for conv layers just couse
 */

//let strNet = fs.readFileSync("xorNet2-10-2.json");

let xorNetz = new Netzwerk(2);

xorNetz.addLayer("dens", {
    "size": 10,
    "actFunc": "relu"
});
xorNetz.addLayer("dens", {
    "size": 2,
    "actFunc": "softmax"
});

xorNetz.setErrorFunction("crossEntropy");
xorNetz.setLossFunction("crossEntropyA");

xorNetz.trainSet([in1, in2, in3, in4], [tar1, tar2, tar3, tar4], {
    "lernRate": 0.01
}, 10000);

console.log("------------------------");

console.log(xorNetz.predict(in1).toString());
console.log(xorNetz.predict(in2).toString());
console.log(xorNetz.predict(in3).toString());
console.log(xorNetz.predict(in4).toString());


throw "Force exit";
/*___________________________________________________________________________________________________________________*/
console.time("Training");

let e = xorNetz.trainSet([in1,in2,in3,in4], [tar1, tar2, tar3, tar4], {
    "lernRate": 0.1
}, 10000);
console.log(e);
console.timeEnd("Training");
console.log(xorNetz.predict(in1).toString());
console.log(xorNetz.predict(in2).toString());
console.log(xorNetz.predict(in3).toString());
console.log(xorNetz.predict(in4).toString());

throw "Force exit";
/*TODO: Implement optimizations */

console.time("Preparing data");
let set = mnist.set(5000, 5000);

let trainingSet = set.training;
let testSet = set.test;

let inps = [];
let targs = [];

for(let i = 0;i < trainingSet.length;i++) {
    inps.push(new Matrix(28,28));
    inps[i].werte = trainingSet[i].input;
    targs.push(new Matrix(1,10));
    targs[i].werte = trainingSet[i].output;
}
console.timeEnd("Preparing data");


console.time("Setting up Network");

let testNetz = new Netzwerk([28,28,1]);

testNetz.setErrorFunction("crossEntropy");
testNetz.setLossFunction("crossEntropyA");

let conv1 = {
    "size": 3,
    "channels": 32,
    "padding": 0,
    "actFunc": "relu"
}

let conv2 = {
    "size": 3,
    "channels": 64,
    "padding": 0,
    "actFunc": "relu"
}

let maxp1 = {
    "size": 2,
    "stride": 2,
    "padding": 0
}

let optsFlatten = {}

let dens1 = {
    "size": 128,
    "actFunc": "relu"
}

let outOpts = {
    "size": 10,
    "actFunc": "softmax"
}

testNetz.addLayer("conv", conv1);
testNetz.addLayer("conv", conv2);
testNetz.addLayer("maxpool", maxp1);
testNetz.addLayer("flatten", optsFlatten);
testNetz.addLayer("dens", dens1)
testNetz.addLayer("dens", outOpts);

console.timeEnd("Setting up Network");

console.log(testNetz.predict(inps[0]).toString());

console.time("Training");
testNetz.trainSet(inps, targs, { lernRate: 0.001}, 1);
console.timeEnd("Training");

