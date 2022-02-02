import { Matrix } from "./netzwerk/Matrix.js";

let matrix = new Matrix(5,5*5);
matrix.randomize();

let filter = new Matrix(5,3*3);
filter.init(0.5);

let mList = [];
let fList = [];
let eList = [];

let e = new Matrix(3,3);
e.init();

for(let i = 0;i < matrix.iDim;i++) {
    mList.push(new Matrix(5,5));
    mList[i].werte = matrix.werte.slice(i*matrix.jDim, (i+1)*matrix.jDim);
    
    fList.push(new Matrix(3,3));
    fList[i].werte = filter.werte.slice(i*filter.jDim, (i+1)*filter.jDim);

    eList.push(mList[i].conv(fList[i]));
    
    e.werte = e.werte.map((x,index)=> {
        let ergebnis = x + eList[i].werte[index];
        return ergebnis;
    })
}

console.log("");

console.log(matrix.convKernel(filter).werte);
console.log(e.werte);