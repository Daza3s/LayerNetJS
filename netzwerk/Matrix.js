class Matrix {
    /**
     * Creates new i * j Matrix
     * @param {Number} iDim 
     * @param {Number} jDim 
     */
    constructor(iDim,jDim) {
        this.werte = new Array(iDim*jDim);      
        this.j = false; //entfernen wenn batch training implementiert
        this.iDim = iDim;
        this.jDim = jDim;
    }

    init(wert = 0) {
        for(let index = 0;index < this.werte.length;index++) {
            this.werte[index] = wert;
        }
    }

    randomize(func) {
        func = func || (() => { return Math.random()*2-1; });
        for(let index = 0;index < this.werte.length;index++) {
            this.werte[index] = func();
        }
    }

    gleicheDim(m) {
        if(this.iDim === m.iDim && this.jDim === m.jDim) return true;

        return false;
    }

    at(i,j, unbounded = false) {
        return this.werte[i * this.jDim + j] ?? (unbounded ? 0 : undefined);
    }

    set(i, j, wert) {
        this.werte[i * this.jDim + j] = wert;
    }

    setSafe(i, j, wert) {
        if(this.inBound(i,j)) {
            this.set(i,j,wert);
        }
    }

    add(m) {
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x+m);
            return erg;
        }
        if(!this.gleicheDim(m)) throw new Error("Wrong scale in matrix manipulation (addition)");
            
        erg.werte = this.werte.map((x,i)=> x + m.werte[i]);

        return erg;
    }

    sub(m) {
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x-m);
            return erg;
        }
        if(!this.gleicheDim(m)) throw new Error("Wrong scale in matrix manipulation (subtraktion)");

        erg.werte = this.werte.map((x,i)=> x - m.werte[i]);

        return erg;
    }

    dot(m) {
        if(m.j) return this.jMult(m);
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x*m);
            return erg;
        }
        if(!this.gleicheDim(m)) {
            throw new Error("Wrong scale in matrix manipulation (dotProdukt)");
        } 

        erg.werte = this.werte.map((x,i)=> x * m.werte[i]);

        return erg;
    }

    div(m) {
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x/m);
            return erg;
        }
        if(!this.gleicheDim(m)) throw new Error("Wrong scale in matrix manipulation (division)");

        erg.werte = this.werte.map((x,i)=> x / m.werte[i]);

        return erg;
    }

    mult(m) {
        if(typeof m === 'number') {
            let erg = new Matrix(this.iDim, this.jDim);
            erg.werte = this.werte.map(x=>x*m);
            return erg;
        }
        try {
            if(this.jDim != m.iDim) throw new Error("Wrong scale in matrix manipulation (crossProdukt)");
        }catch(e) {
            //console.log(m);
            throw e;
        }
        

        let erg = new Matrix(this.iDim, m.jDim);
        for(let i = 0;i < this.iDim;i++) {
            for(let j = 0;j < m.jDim;j++) {
                let sum = 0;
                for(let index = 0;index < this.jDim;index++) {
                    sum += this.at(i,index) * m.at(index, j);
                }
                erg.set(i, j, sum);
            }
        }

        return erg;
    }

    jMult(m) {
        if(this.jDim !== m.werte[0].iDim) throw new Error("Invalid jMult scale!");
        let erg = new Matrix(this.iDim, this.jDim);
        erg.werte = [];

        let zwischen = new Matrix(1,this.jDim);
        
        for(let i = 0;i < this.iDim;i++) {
            zwischen.werte = this.werte.slice(i*this.jDim, i*this.jDim + this.jDim);
            erg.werte.push(...zwischen.mult(m.werte[i]).werte);
        }
        return erg;
    }

    costumMult(m, self) {
        if(self.jDim != m.iDim) throw new Error("Wrong scale in matrix manipulation (crossProdukt)");

        let erg = new Matrix(self.iDim, m.jDim);
        for(let i = 0;i < self.iDim;i++) {
            for(let j = 0;j < m.jDim;j++) {
                let sum = 0;
                for(let index = 0;index < this.jDim;index++) {
                    sum += self.at(i,index) * m.at(index, j);
                }
                erg.set(i, j, sum);
            }
        }
        return erg;
    }

    transpose() {
        let erg = new Matrix(this.jDim, this.iDim);
        for(let i = 0;i < this.iDim;i++) {
            for(let j = 0;j < this.jDim;j++) {
                erg.set( j , erg.jDim - 1 - i , this.at(i,j));
            }
        }
        return erg;
    }

    nest() {
        let erg = [];
        for(let i = 0;i < this.iDim;i++) {
            erg.push([]);
            for(let j = 0;j < this.jDim;j++) {
                erg[i].push(this.at(i,j));
            }
        }
        return erg;
    }

    toString() {
        if(this.j) return "jMatrix: " + this.werte.toString();
        let erg = '';
        let nested = this.nest();
        for(let i = 0;i < nested.length;i++) {
            erg += JSON.stringify(nested[i]) + '\n';
        }        
        return erg.slice(0,-1);
    }

    toJSON() {
        let erg = {
            "json": true,
            "iDim": this.iDim,
            "jDim": this.jDim,
            "werte": this.werte,
            "j": this.j
        }
        return erg;
    }

    fromJSON(json) {
        this.iDim = json.iDim;
        this.jDim = json.jDim;
        this.werte = json.werte;
        this.j = json.j;
    }

    maxNorm() {
        let max = 0;

        for(let i = 0;i < this.iDim;i++) {
            let sum = 0;
            for(let j = 0;j < this.jDim;j++) {
                sum += this.at(i,j)**2;
            }

            sum = Math.sqrt(sum);

            if(sum > max) max = sum;
        }

        return max;
    }

    sum(func) {
        func = func || ((a)=>{ return a });
        let erg = 0;

        for(let i = 0;i < this.werte.length;i++) {
            erg += func(this.werte[i]);
        }
        
        return erg;
    }

    /**
     * 
     * @param {Matrix} filter 
     * @param {number} stride 
     * @param {number} padding
     * @returns {Matrix} 
     */
    conv(filter, stride = 1, padding = 0) {
        let ergI = (this.iDim + 2*padding - filter.iDim) / stride + 1;
        let ergJ = (this.jDim + 2*padding - filter.jDim) / stride + 1;
        let erg = new Matrix( ergI , ergJ );

        for(let i = 0 - padding;i < ergI + padding;i++) {
            for(let j = 0 - padding;j < ergJ + padding;j++) {
                let e_ij = 0;

                for(let filterI = 0;filterI < filter.iDim;filterI++) {
                    for(let filterJ = 0;filterJ < filter.jDim;filterJ++) {
                        e_ij += filter.at(filterI,filterJ, true) * this.at(filterI + i*stride, filterJ + j*stride, true);
                    }
                }

                erg.set(i,j, e_ij);
            }
        }

        return erg;
    }

    upscale(filter, stride = 1, padding = 0) {
        let ergI = (this.iDim - 1) * stride + filter.iDim - 2 * padding;
        let ergJ = (this.jDim - 1) * stride + filter.jDim - 2 * padding;

        let erg = new Matrix( ergI , ergJ );
        erg.init();
        for(let i = 0;i < this.iDim; i++) {
            for(let j = 0;j < this.jDim; j++) {
                let zwischen = filter.mult(this.at(i,j));

                for(let zI = 0;zI < zwischen.iDim;zI++) {
                    for(let zJ = 0;zJ < zwischen.jDim;zJ++) {
                        erg.setSafe( zI+(i*stride)-padding , zJ+(j*stride)-padding , erg.at( zI+(i*stride)-padding , zJ+(j*stride)-padding , true ) + zwischen.at(zI,zJ));
                    }
                }

            }
        }
        return erg;
    }

    /**
     * Checks if coordinates are valid
     * @param {number} i 
     * @param {number} j 
     * @returns {bool}
     */
    inBound(i,j) {
        if(i < 0 || j < 0 || i >= this.iDim || j >= this.jDim) return false;
        return true;
    }

    /**
     * Sums over i Dimension
     * @returns {Matrix}
     */
    collapse() {
        let erg = new Matrix(1,this.jDim);
        erg.init();
        for(let i = 0;i < this.iDim;i++) {
            for(let j = 0;j < this.jDim;j++) {
                erg.set(0,j,erg.at(0,j)+this.at(i,j));
            }
        }
        return erg;
    }

    kasten(i, j, filter) {
        let erg = new Matrix(filter.iDim, filter.jDim);
        erg.werte = filter.werte.map((x,index)=> {
            let offSet = index % filter.iDim;
            return x * this.at(i + (index-offSet)/filter.iDim, j + offSet);
        });
        return erg;
    }

    kasten3DSum(i, j, filter, unbounded = false) {
        let erg = [];
        let d1I, d1J = Math.sqrt(this.jDim);
        for(let index = 0;index < this.iDim;index++) {
            let arr = filter.werte.map((x,fIndex)=> {
                let offSet = fIndex % filter.iDim;
                return x * this.at(index, (j + offSet) + (fIndex - offSet + i) * d1J, unbounded);
            });
            let sum = arr.reduce((a,b)=> a+b);
            erg.push(sum);
        }
        return erg.reduce((a,b)=> a+b);
    }
    /**
     * TODO: FIX
     * @param {*} filter 
     * @param {*} stride 
     * @param {*} padding 
     * @returns 
     */
    convKernel(filter, stride = 1, padding = 0) {
        //--
        let filterI, filterJ, ownI, ownJ;
        filterI = filterJ = Math.sqrt(filter.jDim);
        ownI = ownJ = Math.sqrt(this.jDim);

        let ergI = (ownI + 2*padding - filterI) / stride + 1;
        let ergJ = (ownJ + 2*padding - filterJ) / stride + 1;

        let erg = new Matrix(1, ergI*ergJ);
        //loop over i and j of pic
        for(let i = -padding;i < ergI+padding;i += stride) {
            for(let j = -padding;j < ergJ+padding;j += stride) {
                let sum = this.kasten3DSum(i, j, filter, true);
                erg.werte[i * ergJ + j] = sum;
            }
        }
        return erg;
        //--
    }
}


export { Matrix }