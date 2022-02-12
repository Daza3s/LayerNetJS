import { Matrix } from "./netzwerk/Matrix.js";

let nM = new Matrix(5,5);
nM.init(1);

console.log(nM.toString());

console.log("");

let nF = new Matrix(3,3);
nF.init(0.5);

let nC = nM.conv(nF, 1, 1);

console.log(nC.toString());



let mPacked = new Matrix(3,5*5);
let fPacked = new Matrix(3,3*3);

mPacked.packed(5,5);
fPacked.packed(3,3);

mPacked.init(1);
fPacked.init(.5);

/**FIX PADDING */
let pErg = mPacked.convKernel(fPacked, 1, 1);

console.log(mPacked.werte);
console.log(pErg.werte);

console.log(pErg.werte.map((x,i)=>x-nC.werte[i]))