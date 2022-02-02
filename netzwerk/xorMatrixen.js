import { Matrix } from "./Matrix.js";

let in1 = new Matrix(1,2);
let in2 = new Matrix(1,2);
let in3 = new Matrix(1,2);
let in4 = new Matrix(1,2);

in1.werte = [0,0];
in2.werte = [0,1];
in3.werte = [1,0];
in4.werte = [1,1];

let tar1 = new Matrix(1,2);
let tar2 = new Matrix(1,2);
let tar3 = new Matrix(1,2);
let tar4 = new Matrix(1,2);

tar1.werte = [1,0];
tar2.werte = [0,1];
tar3.werte = [0,1];
tar4.werte = [1,0];

let batchIn = new Matrix(4,2);
batchIn.werte = [0,0,0,1,1,0,1,1];

let batchTar = new Matrix(4, 2);
batchTar.werte = [1,0,0,1,0,1,1,0];

export { in1, in2, in3, in4, tar1, tar2, tar3, tar4, batchIn, batchTar }