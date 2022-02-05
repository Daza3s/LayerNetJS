import { Matrix } from "./netzwerk/Matrix.js";

let mPacked = new Matrix(4,5*5);
let fPacked = new Matrix(4,3*3);

mPacked.packed(5,5);
fPacked.packed(3,3);

mPacked.init(1);
fPacked.init(.5);

/**FIX PADDING AND STRIDE */
let pErg = mPacked.convKernel(fPacked, 1, 1);

console.log(pErg);

/**FIX NON SQUARE CONVOLUTION */