#![allow(non_snake_case)]

// Macro for defining the following types:
// - ThetaPoint: An element of the level-2 theta structure, which is encoded by
// four projective coordinates: (Fq : Fq : Fq : Fq)
//
// - ThetaStructure: The parent of ThetaPoint, the identity point is the theta
// null point of type ThetaPoint. For arithmetic, this type also has an
// arithmetic precom. which can be reused for both doublings and isogenies.
//
// - product_isogeny: an implementation of an isogeny between elliptic products
// (E1 x E2) of type EllipticProduct = (Curve, Curve) with a kernel of type
// (CouplePoint, CouplePoint) where each CouplePoint represents a pair of points
// P1, P2 on E1 and E2 respectively

// Macro expectations:
// Fq      type of field element Fp^2
// Curve   type of elliptic curve in Montgomery model
// Point   type of point on a montgomery curve
// EllipticProduct    type of E1 x E2
// CouplePoint        type of point on E1 x E2
macro_rules! define_theta_structure {
    () => {
        use std::fmt;

        /// Given four elements of Fq, compute the hadamard transform using recursive
        /// addition.
        /// Cost: 8a
        #[inline(always)]
        fn to_hadamard(X: &Fq, Y: &Fq, Z: &Fq, T: &Fq) -> (Fq, Fq, Fq, Fq) {
            let t1 = X + Y;
            let t2 = X - Y;
            let t3 = Z + T;
            let t4 = Z - T;

            let A = &t1 + &t3;
            let B = &t2 + &t4;
            let C = &t1 - &t3;
            let D = &t2 - &t4;
            (A, B, C, D)
        }

        /// Given four elements of Fq, first square each coordinate
        /// Cost: 4S
        #[inline(always)]
        fn to_squared_coords(X: &Fq, Y: &Fq, Z: &Fq, T: &Fq) -> (Fq, Fq, Fq, Fq) {
            let XX = X.square();
            let YY = Y.square();
            let ZZ = Z.square();
            let TT = T.square();

            (XX, YY, ZZ, TT)
        }

        /// Given four elements of Fq, first square each coordinate and
        /// then compute the hadamard transform
        /// Cost: 4S, 8a
        #[inline(always)]
        fn to_squared_theta(X: &Fq, Y: &Fq, Z: &Fq, T: &Fq) -> (Fq, Fq, Fq, Fq) {
            let (XX, YY, ZZ, TT) = to_squared_coords(X, Y, Z, T);
            to_hadamard(&XX, &YY, &ZZ, &TT)
        }

        // ========================================================
        // Functions for working with ThetaPoints
        // ========================================================

        /// Theta Point Struct
        #[derive(Clone, Copy, Debug)]
        pub struct ThetaPoint {
            X: Fq,
            Y: Fq,
            Z: Fq,
            T: Fq,
        }

        impl ThetaPoint {
            /// Use for initalisation, probably stupid, or at least should have
            /// a different name!
            pub const ZERO: Self = Self {
                X: Fq::ZERO,
                Y: Fq::ZERO,
                Z: Fq::ZERO,
                T: Fq::ZERO,
            };

            /// Compile time, create a new theta point from Fq elements
            pub const fn new(X: &Fq, Y: &Fq, Z: &Fq, T: &Fq) -> Self {
                Self {
                    X: *X,
                    Y: *Y,
                    Z: *Z,
                    T: *T,
                }
            }

            /// Create a new theta point from Fq elements
            pub fn from_coords(X: &Fq, Y: &Fq, Z: &Fq, T: &Fq) -> Self {
                Self {
                    X: *X,
                    Y: *Y,
                    Z: *Z,
                    T: *T,
                }
            }

            /// Recover the coordinates of the element
            pub fn coords(self) -> (Fq, Fq, Fq, Fq) {
                (self.X, self.Y, self.Z, self.T)
            }

            /// Recover the coordinates of the element
            pub fn list(self) -> [Fq; 4] {
                [self.X, self.Y, self.Z, self.T]
            }

            /// Compute the Hadamard transform of the point's coordinates
            pub fn hadamard(self) -> (Fq, Fq, Fq, Fq) {
                to_hadamard(&self.X, &self.Y, &self.Z, &self.T)
            }

            /// Square each of the point's coordinates and then
            /// compute the hadamard transform
            pub fn squared_theta(self) -> (Fq, Fq, Fq, Fq) {
                to_squared_theta(&self.X, &self.Y, &self.Z, &self.T)
            }
        }

        /// For debugging, pretty print out the coordinates of a point
        impl fmt::Display for ThetaPoint {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}\n{}\n{}\n{}\n", self.X, self.Y, self.Z, self.T)
            }
        }

        // ========================================================
        // Functions for working with ThetaStructures
        // ========================================================

        /// Theta Structure
        #[derive(Clone, Copy, Debug)]
        pub struct ThetaStructure {
            null_point: ThetaPoint,
            arithmetic_precom: [Fq; 8],
        }

        impl ThetaStructure {
            /// Given the coordinates of a null point, create a null point and
            /// precompute 8 Fp2 elements which are used for doubling and isogeny
            /// computations.
            pub fn new_from_coords(X: &Fq, Z: &Fq, U: &Fq, V: &Fq) -> Self {
                let null_point = ThetaPoint::new(X, Z, U, V);
                Self {
                    null_point,
                    arithmetic_precom: ThetaStructure::precomputation(&null_point),
                }
            }

            /// Given a null point, store the null point and precompute 8 Fp2
            /// elements which are used for doubling and isogeny computations.
            pub fn new_from_point(null_point: &ThetaPoint) -> Self {
                Self {
                    null_point: *null_point,
                    arithmetic_precom: ThetaStructure::precomputation(null_point),
                }
            }

            /// Return the null point of the ThetaStructure
            pub fn null_point(self) -> ThetaPoint {
                self.null_point
            }

            /// For doubling and also computing isogenies, we need the following
            /// constants, which we can precompute once for each ThetaStructure.
            /// Cost: 14M + 5S
            #[inline]
            pub fn precomputation(O0: &ThetaPoint) -> [Fq; 8] {
                let (a, b, c, d) = O0.coords();
                let (aa, bb, cc, dd) = to_squared_coords(&a, &b, &c, &d);

                // Compute projectively a/b = a^2*c*d, etc.
                let ab = &a * &b;
                let cd = &c * &d;
                let x0 = &ab * &cd;
                let y0 = &aa * &cd;
                let aab = &aa * b;
                let z0 = &aab * &d;
                let t0 = &aab * &c;

                // Compute projectively A^2/B^2 = A^4*C^2*D^2, etc.
                let (AA, BB, CC, DD) = to_hadamard(&aa, &bb, &cc, &dd);
                let A4 = AA.square();
                let AABB = &AA * &BB;
                let CCDD = &CC * &DD;
                let X0 = &AABB * &CCDD;
                let Y0 = &A4 * &CCDD;
                let A4BB = &A4 * &BB;
                let Z0 = &A4BB * &DD;
                let T0 = &A4BB * &CC;

                [x0, y0, z0, t0, X0, Y0, Z0, T0]
            }

            /// Given a point P, compute it's double [2]P in place.
            /// Cost 8S + 8M
            #[inline(always)]
            pub fn set_double_self(self, P: &mut ThetaPoint) {
                let (mut xp, mut yp, mut zp, mut tp) = P.squared_theta();

                // Compute temp. coordinates, 8S and 3M
                xp = &self.arithmetic_precom[4] * &xp.square();
                yp = &self.arithmetic_precom[5] * &yp.square();
                zp = &self.arithmetic_precom[6] * &zp.square();
                tp = &self.arithmetic_precom[7] * &tp.square();

                // Compute the final coordinates, 3M
                let (mut X, mut Y, mut Z, mut T) = to_hadamard(&xp, &yp, &zp, &tp);
                X *= &self.arithmetic_precom[0];
                Y *= &self.arithmetic_precom[1];
                Z *= &self.arithmetic_precom[2];
                T *= &self.arithmetic_precom[3];

                P.X = X;
                P.Y = Y;
                P.Z = Z;
                P.T = T;
            }

            /// Compute [2] * self
            #[inline]
            pub fn double_point(self, P: &ThetaPoint) -> ThetaPoint {
                let mut P2 = *P;
                self.set_double_self(&mut P2);
                P2
            }

            /// Compute [2^n] * self
            #[inline]
            pub fn double_iter(self, P: &ThetaPoint, n: usize) -> ThetaPoint {
                let mut R = *P;
                for _ in 0..n {
                    self.set_double_self(&mut R)
                }
                R
            }
        }





        /// Apply the base change described by M on a ThetaPoint in-place
        /// Cost: 16M
        #[inline]
        fn apply_base_change(P: &mut ThetaPoint, M: [Fq; 16]) {
            let (x, y, z, t) = P.coords();
            P.X = &M[0] * &x + &M[1] * &y + &M[2] * &z + &M[3] * &t;
            P.Y = &M[4] * &x + &M[5] * &y + &M[6] * &z + &M[7] * &t;
            P.Z = &M[8] * &x + &M[9] * &y + &M[10] * &z + &M[11] * &t;
            P.T = &M[12] * &x + &M[13] * &y + &M[14] * &z + &M[15] * &t;
        }



        // ========================================================
        // Compting the gluing (2,2)-isogeny from a product of
        // elliptic curves to a level 2 theta structure
        //
        // The following functions are derived from the Superglue paper
        // ========================================================








        /// Given two points T1 and T2 in E[4] that define a symplectic basis, compute the
        /// symetric elements g1, g2 and g3 that correspond to the action of T1, T2 and T1+T2
        /// these symetric elements all have the same denominator delta.
        /// This is a subroutine of the Compute_Superglue_Coefficients function.
    
        fn Compute_symetric_element(T1: &Point, T2: &Point) -> (u8, Fq, [Fq; 3],[Fq; 3],[Fq; 3]) {

            let (x1, z1) = T1.to_xz();
            let (x2, z2) = T2.to_xz();

            let mut pos: u8 = 255;
            let mut r1 = &x1 + &z1;
            let mut s1 = &x1 - &z1;
            let mut r2 = &x2 + &z2;
            let mut s2 = &x2 - &z2;

            if s1.iszero() != 0 {
                pos = 0;
            }
            if r1.iszero()!= 0 {
                pos = 1;
            }
            if s2.iszero()!= 0 {
                pos = 2;
            }
            if r2.iszero()!= 0 {
                pos = 3;
            }

            s2 = &r1 * &s2; // s2 = (x1 + z1)(x2 - z2)
            r1 = Fq::ZETA * &s2; // r1 = i(x1 + z1)(x2 - z2)
            r2 = &r2 * &s1; // r2 = (x1 - z1)(x2 + z2)
            s1 = &r1 + &r2; 
            s2 = &r1 - &r2; 

            if s1.iszero()!= 0 {
                pos = 4;
            }
            if s2.iszero()!= 0 {
                pos = 5;
            }

            match pos {

                0 => {
                    s1 = x2.square();
                    s2 = z2.square();
                    let delta = &s1 - &s2; // delta = x2^2 - z2^2
                    r1 = &s1 + &s2; // r1 = x2^2 + z2^2
                    r2 = &x2 * &z2; // r2 = x2z2
                    r2 = &r2 + &r2; // r2 = 2x2z2

                    // g1 = (0,-d),(-d,0)
                    let g1 = [Fq::ZERO, delta, delta];
                    let g2 = [r1, -&r2, r2];
                    let g3 = [Fq::ZETA * &r2, Fq::MINUS_ZETA * &r1, Fq::ZETA * &r1];

                    (pos, delta, g1, g2, g3)

                }

                1 => {
                    s1 = x2.square();
                    s2 = z2.square();
                    let delta = &s1 - &s2; // delta = x2^2 - z2^2
                    r1 = &s1 + &s2; // r1 = x2^2 + z2^2
                    r2 = &x2 * &z2; // r2 = x2z2
                    r2 = &r2 + &r2; // r2 = 2x2z2

                    let g1 = [Fq::ZERO, -&delta, -&delta];
                    let g2 = [r1, -&r2, r2];
                    let g3 = [Fq::MINUS_ZETA * &r2, Fq::ZETA * &r1, Fq::MINUS_ZETA * &r1];

                    (pos,delta, g1, g2, g3)
                }

                2 => {
                    s1 = x1.square();
                    s2 = z1.square();
                    let delta = &s1 - &s2; // delta = x1^2 - z1^2
                    r1 = &s1 + &s2; // r1 = x1^2 + z1^2
                    r2 = &x1 * &z1; // r2 = x1z1
                    r2 = &r2 + &r2; // r2 = 2x1z1

                    let g1 = [r1, -&r2, r2];
                    let g2 = [Fq::ZERO, delta, delta];
                    let g3 = [Fq::MINUS_ZETA * &r2, Fq::ZETA * &r1, Fq::MINUS_ZETA * &r1];

                    (pos,delta, g1, g2, g3)
                }

                3 => {
                    s1 = x1.square();
                    s2 = z1.square();
                    let delta = &s1 - &s2; // delta = x1^2 - z1^2
                    r1 = &s1 + &s2; // r1 = x1^2 + z1^2
                    r2 = &x1 * &z1; // r2 = x1z1
                    r2 = &r2 + &r2; // r2 = 2x1z1

                    let g1 = [r1, -&r2, r2];
                    let g2 = [Fq::ZERO, -&delta, -&delta];
                    let g3 = [Fq::ZETA * &r2, Fq::MINUS_ZETA * &r1, Fq::ZETA * &r1];

                    (pos,delta, g1, g2, g3)
                }

                4 => {
                    s1 = x1.square();
                    s2 = z1.square();
                    let delta = &s1 - &s2; // delta = x1^2 - z1^2
                    r1 = &s1 + &s2; // r1 = x1^2 + z1^2
                    r2 = &x1 * &z1; // r2 = x1z1
                    r2 = &r2 + &r2; // r2 = 2x1z1

                    let g1 = [r1, -&r2, r2];
                    let g2 = [Fq::MINUS_ZETA * &r2, Fq::ZETA * &r1, Fq::MINUS_ZETA * &r1];
                    let g3 = [Fq::ZERO, -&delta, -&delta];

                    (pos,delta, g1, g2, g3)
                }


                5 => {
                    s1 = x1.square();
                    s2 = z1.square();
                    let delta = &s1 - &s2; // delta = x1^2 - z1^2
                    r1 = &s1 + &s2; // r1 = x1^2 + z1^2
                    r2 = &x1 * &z1; // r2 = x1z1
                    r2 = &r2 + &r2; // r2 = 2x1z1

                    let g1 = [r1, -&r2, r2];
                    let g2 = [Fq::ZETA * &r2, Fq::MINUS_ZETA * &r1, Fq::ZETA * &r1];
                    let g3 = [Fq::ZERO, delta, delta];

                    (pos, delta, g1, g2, g3)
                }
                
                _ => {
                    panic!("Invalid position for Compute_symetric_element");
                }
            }
        }



        /// Given the four torsion below the isogeny kernel, compute the
        /// right superglue coefficients.
        /// Input is expected to be K1 = (P1, P2), K2 = (Q1, Q2) in E1 x E2
        /// Inside (E1 x E2)[4].
        fn Compute_Superglue_Coefficients(
            P1P2: &CouplePoint,
            Q1Q2: &CouplePoint,
        ) -> (u8, bool, [Fq; 4], [Fq; 2], [Fq; 2], [Fq; 2]) {
            // First compute the submatrices from each point
            let (P1, P2) = P1P2.points();
            let (Q1, Q2) = Q1Q2.points();

            let (pos1, delta1, g0_1, g1_1, g2_1) = Compute_symetric_element(&P1, &Q1);
            let (pos2, delta2, g0_2, g1_2, g2_2) = Compute_symetric_element(&P2, &Q2);
            let delta = &delta1 * &delta2;

            let pos: u8 = 3*(pos1 >> 1) + (pos2 >> 1);

            
            let case: u8;
            
            match pos {
                0 => {case = 1}     //Special case 
                4 => {case = 2}     //Special case
                8 => {case = 3}     //Special case
                _ => {case = 0}     //Standard case
            }


            let mut sgl_val = Fq::ZERO;
            
            let mut a = delta;
            let mut b = Fq::ZERO;
            let mut c = Fq::ZERO;
            let mut d = Fq::ZERO;

            // T1
            a += &g0_1[0] * &g0_2[0];
            b += &g0_1[0] * &g0_2[1];
            c += &g0_1[1] * &g0_2[0];
            d += &g0_1[1] * &g0_2[1];

            // T2
            a += &g1_1[0] * &g1_2[0];
            b += &g1_1[0] * &g1_2[1];
            c += &g1_1[1] * &g1_2[0];
            d += &g1_1[1] * &g1_2[1];

            // T1+T2
            a -= &g2_1[0] * &g2_2[0];
            b -= &g2_1[0] * &g2_2[1];
            c -= &g2_1[1] * &g2_2[0];
            d -= &g2_1[1] * &g2_2[1];

            // Now we act by (0, Q2)
            let a1 = delta1*(&g1_2[0] * &a + &g1_2[2] * &b);
            if case == 1 || case == 3 {
                sgl_val = delta1 * (&g1_2[1] * &a - &g1_2[0] * &b);
            }


            // Now we act by (P1, 0)
            let  a2 = delta2 * (&g0_1[0] * &a + &g0_1[2] * &c);
            if case == 2 {
                sgl_val = delta2 * (&g0_1[0] * &b + &g0_1[2] * &d);
            }
            

            // Refactoring
            a *= delta; 
            b *= delta; 
            c *= delta; 
            d *= delta; 

            let mut dual_theta_null_square = [Fq::ZERO, a1.square(),a2.square(),a.square()];
            dual_theta_null_square[0] = &dual_theta_null_square[1] + &dual_theta_null_square[2];
            dual_theta_null_square[1] = &dual_theta_null_square[3] - &dual_theta_null_square[1];
            dual_theta_null_square[2] = &dual_theta_null_square[3] - &dual_theta_null_square[2];
            dual_theta_null_square[3] = Fq::ZERO;

            let mut tmp1;
            let mut tmp2;
            let mut tmp3;

            match case {
                1 => {
                    tmp1 = &a * &b;
                    tmp2 = &a1 * &sgl_val;
                    let element_01 = [&tmp1 + &tmp2, &tmp1 - &tmp2];

                    tmp3 = &a * &c;
                    tmp1 -= &tmp3;                
                    let sign_flip = (tmp1.iszero() == 0);

                    // Useless application to be constant time
                    let element_02 = [&tmp1 + &tmp2, &tmp1 - &tmp2];
                    let element_03 = [tmp1, tmp2];

                    return (pos,sign_flip, dual_theta_null_square, element_01, element_02, element_03);
                }
                2 => {
                    tmp1 = &a * &b;
                    tmp2 = &a2 * &sgl_val;
                    let element_01 = [&tmp1 + &tmp2, &tmp1 - &tmp2];
                    
                    // Set the zeros properly
                    tmp3 = &a * &c;
                    tmp1 -= &tmp3;                
                    let sign_flip = (tmp1.iszero() == 0);

                    // Useless application to be constant time
                    let element_02 = [&tmp1 + &tmp2, &tmp1 - &tmp2];
                    let element_03 = [tmp1, tmp2];

                    return (pos,sign_flip, dual_theta_null_square, element_01, element_02, element_03);
                }
                3 => {
                    tmp1 = &a * &b;
                    tmp2 = &a1 * &sgl_val;
                    let element_01 = [&tmp1 + &tmp2, &tmp1 - &tmp2];

                    tmp3 = &a * &c;
                    tmp1 -= &tmp3;                
                    let sign_flip = (tmp1.iszero() == 0);

                    // Useless application to be constant time
                    let element_02 = [&tmp1 + &tmp2, &tmp1 - &tmp2];
                    let element_03 = [tmp1, tmp2];

                    return (pos,sign_flip, dual_theta_null_square, element_01, element_02, element_03);
                }
                _ => {
                    tmp1 = &a * &b;
                    tmp2 = &c * &d;
                    let element_01 = [&tmp1 + &tmp2, &tmp1 - &tmp2];

                    let sign_flip = false;

                    tmp3 = &a * &c;
                    tmp2 = &b * &d;
                    let element_02 = [&tmp3 + &tmp2, &tmp3 - &tmp2];

                    tmp1 = &a * &d;
                    tmp3 = &b * &c;
                    let element_03 = [&tmp1 + &tmp3, &tmp1 - &tmp3];

                    return (pos,sign_flip, dual_theta_null_square, element_01, element_02, element_03);


                }
            }

        }



        /// Given Take P and Q in E distinct, return three components u,v and w in Fp2 such
        /// that the xz coordinates of P+Q are (u-v:w) and of P-Q are (u+v:w)
        

        fn Points_to_barycentric_coordinates(E1: Curve, P: Point, Q: Point) -> [Fq;3] {

            let mut t = [Fq::ZERO; 7];

            t[0] = &P.get_Z() * &Q.get_Z();        // t0 = z1z2
            t[1] = &P.get_X() * &Q.get_Z();        // t1 = x1z2
            t[2] = &P.get_Z() * &Q.get_X();        // t2 = z1x2
            t[3] = &P.get_Y() * &Q.get_Z();        // t3 = y1z2
            t[4] = &P.get_Z() * &Q.get_Y();        // t4 = z1y2

            t[5] = &t[3] * &t[4];    // t5 = y1y2z1z2
            t[5] = &t[5] * &t[0];    // t5 = y1y2z1^2z2^2

            t[6] = &E1.get_A() * &t[0];    // t6 = Az1z2
            t[1] = &t[1] + &t[2];    // t1 = z1x2 + x1z2
            t[6] = &t[6] + &t[1];    // t6 = Az1z2 + z1x2 + x1z2
            t[2] = &t[2] + &t[2];    // t2 = 2x1z2 

            t[1] = &t[1] - &t[2];    // t1 = z1x2 - x1z2 
            t[1] = t[1].square();   // t1 = (z1x2 - x1z2)^2
            t[6] = &t[6] * &t[1];    // t6 = (Az1z2 + z1x2 + x1z2)(z1x2 - x1z2)^2

            t[3] =  t[3].square();   // t3 = y1^2z2^2
            t[4] =  t[4].square();   // t4 = y2^2z1^2
            t[3] = &t[3] + &t[4];   // t3 = y1^2z2^2 + y2^2z1^2
            t[3] = &t[3] * &t[0];    // t3 = z1z2(y1^2z2^2 + y2^2z1^2)

            [&t[3]-&t[6],&t[5]+&t[5],&t[0] * &t[1]]
        
        }



        fn Superglue_Evaluation( 
            bary1: [Fq;3],
            bary2: [Fq;3],
            position: u8,
            sign_flip : bool,
            dual_theta_null_square: [Fq; 4],
            element_01: [Fq; 2],
            element_02: [Fq; 2],
            element_03: [Fq; 2]
            ) -> ThetaPoint 
        {

            let mut uuvv_p_ww1: Fq;
            let uuvv_m_ww1: Fq;
            let mut uuvv_p_ww2: Fq;
            let uuvv_m_ww2: Fq;
            let mut uw1: Fq;
            let mut uw2: Fq;
            let mut vw12: Fq;
            let mut tmp1: Fq;
            let mut tmp2: Fq;
            let mut tmp3 = Fq::ZERO;
            
            if sign_flip {
                tmp3 = &tmp3 - &bary1[2];
            } else {
                tmp3 = &bary1[2] - &tmp3;
            }
            
            uuvv_p_ww1 = &bary1[0] + &bary1[1];
            tmp1 = &bary1[0] - &bary1[1];
            uuvv_p_ww1 = &uuvv_p_ww1 * &tmp1; // uuvv1 = u1^2 - v1^2

            uuvv_p_ww2 = &bary2[0] + &bary2[1];
            tmp1 = &bary2[0] - &bary2[1];
            uuvv_p_ww2 = &uuvv_p_ww2 * &tmp1; // uuvv2 = u1^2 - v1^2

            tmp1 = bary1[2].square(); // ww1 = w1^2
            tmp2 = bary2[2].square(); // ww2 = w1^2

            uuvv_m_ww1 = &uuvv_p_ww1 - &tmp1; // uuvv_m_ww1 = u1^2 - w1^2
            uuvv_p_ww1 = &uuvv_p_ww1 + &tmp1; // uuvv_p_ww1 = u1^2 + w1^2
            uuvv_m_ww2 = &uuvv_p_ww2 - &tmp2; // uuvv_m_ww2 = u2^2 - w2^2
            uuvv_p_ww2 = &uuvv_p_ww2 + &tmp2; // uuvv_p_ww2 = u2^2 + w2^2

            uw1 = &bary1[0] * &tmp3;         // uw1 = ±u1w1
            uw1 = &uw1 + &uw1;               // uw1 = ±2u1w1
            uw2 = &bary2[0] * &bary2[2];     // uw2 = u2w2
            uw2 = &uw2 + &uw2;               // uw2 = 2u2w2

            vw12 = &bary1[1] * &bary2[1];     // vw12 = v1v2
            tmp1 = &tmp3 * &bary2[2];         // tmp1 = ±w1w2
            vw12 = &vw12 * &tmp1;             // vw12 = ±v1v2w1w2
            vw12 = &vw12 + &vw12;             // vw12 = ±2v1v2w1w2
            vw12 = &vw12 + &vw12;             // vw12 = ±4v1v2w1w2


            match position {

                1 => {

                    // Compute first element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[0];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_02[0];
                    tmp2 = &uw2 * &element_03[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let X = &tmp3 + &tmp1;

                    // Compute second element
                    tmp1 = &uuvv_p_ww1 * &dual_theta_null_square[1];
                    tmp2 = &uw1 * &element_02[1];
                    tmp1 = &tmp1 * &uuvv_m_ww2;
                    tmp2 = &tmp2 * &uuvv_m_ww2;
                    tmp1 = &tmp1 + &tmp2;
                    let Y = Fq::ZERO + &tmp1;


                    // Compute third element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[2];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 = &tmp1 * &uuvv_m_ww1;
                    tmp2 = &tmp2 * &uuvv_m_ww1;
                    let Z = &tmp1 + &tmp2;

                    // Compute the last point.
                    let T = &vw12 * &element_03[1];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)


                }
                2 => {

                    // Compute second element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[1];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_02[0];
                    tmp2 = &uw2 * &element_03[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Y = &tmp3 + &tmp1;

                    // Compute first element
                    tmp1 = &uuvv_p_ww1 * &dual_theta_null_square[0];
                    tmp2 = &uw1 * &element_02[1];

                    tmp1 = &tmp1 * &uuvv_m_ww2;
                    tmp2 = &tmp2 * &uuvv_m_ww2;
                    tmp1 = &tmp1 + &tmp2;
                    let X = Fq::ZERO + &tmp1;


                    // Compute third element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[2];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 = &tmp1 * &uuvv_m_ww1;
                    tmp2 = &tmp2 * &uuvv_m_ww1;
                    let Z = &tmp1 + &tmp2;

                    // Compute the last point.
                    let T = &vw12 * &element_03[1];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)


                }

                3 => {

                // Compute first element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[0];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_02[0];
                    tmp2 = &uw2 * &element_03[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let X = &tmp3 + &tmp1;

                    // Compute third element
                    tmp1 = &uuvv_p_ww1 * &dual_theta_null_square[2];
                    tmp2 = &uw1 * &element_02[1];
                    tmp1 = &tmp1 * &uuvv_m_ww2;
                    tmp2 = &tmp2 * &uuvv_m_ww2;
                    tmp1 = &tmp1 + &tmp2;
                    let Z = Fq::ZERO + &tmp1;

                    // Compute second element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[1];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 = &tmp1 * &uuvv_m_ww1;
                    tmp2 = &tmp2 * &uuvv_m_ww1;
                    let Y = &tmp1 + &tmp2;

                    // Compute the last point.
                    let T = &vw12 * &element_03[1];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)


                }

                5 => {

                // Compute third element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[2];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_02[0];
                    tmp2 = &uw2 * &element_03[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Z = &tmp3 + &tmp1;

                    // Compute first element
                    tmp1 = &uuvv_p_ww1 * &dual_theta_null_square[0];
                    tmp2 = &uw1 * &element_02[1];
                    tmp1 = &tmp1 * &uuvv_m_ww2;
                    tmp2 = &tmp2 * &uuvv_m_ww2;
                    tmp1 = &tmp1 + &tmp2;
                    let X = Fq::ZERO + &tmp1;

                    // Compute second element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[1];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 = &tmp1 * &uuvv_m_ww1;
                    tmp2 = &tmp2 * &uuvv_m_ww1;
                    let Y = &tmp1 + &tmp2;

                    // Compute the last point.
                    let T = &vw12 * &element_03[1];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)


                }

                6 => {

                // Compute second element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[1];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_02[0];
                    tmp2 = &uw2 * &element_03[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Y = &tmp3 + &tmp1;

                    // Compute third element
                    tmp1 = &uuvv_p_ww1 * &dual_theta_null_square[2];
                    tmp2 = &uw1 * &element_02[1];
                    tmp1 = &tmp1 * &uuvv_m_ww2;
                    tmp2 = &tmp2 * &uuvv_m_ww2;
                    tmp1 = &tmp1 + &tmp2;
                    let Z = Fq::ZERO + &tmp1;

                    // Compute first element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[0];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 = &tmp1 * &uuvv_m_ww1;
                    tmp2 = &tmp2 * &uuvv_m_ww1;
                    let X = &tmp1 + &tmp2;

                    // Compute the last point.
                    let T = &vw12 * &element_03[1];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)


                }

                7 => {

                // Compute third element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[2];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_02[0];
                    tmp2 = &uw2 * &element_03[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Z = &tmp3 + &tmp1;

                    // Compute second element
                    tmp1 = &uuvv_p_ww1 * &dual_theta_null_square[1];
                    tmp2 = &uw1 * &element_02[1];
                    tmp1 = &tmp1 * &uuvv_m_ww2;
                    tmp2 = &tmp2 * &uuvv_m_ww2;
                    tmp1 = &tmp1 + &tmp2;
                    let Y = Fq::ZERO + &tmp1;

                    // Compute first element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[0];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 = &tmp1 * &uuvv_m_ww1;
                    tmp2 = &tmp2 * &uuvv_m_ww1;
                    let X = &tmp1 + &tmp2;

                    // Compute the last point.
                    let T = &vw12 * &element_03[1];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)


                }

                0 => {

                    // Compute first element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[0];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_01[1];
                    tmp2 = &uw2 * &dual_theta_null_square[1];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let X = &tmp3 + &tmp1;

                    // Compute second element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[1];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_01[0];
                    tmp2 = &uw2 * &dual_theta_null_square[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Y = &tmp3 + &tmp1;

                    // Compute third element
                    tmp1 = &uuvv_m_ww2 * &uuvv_m_ww1;
                    let Z = &tmp1 * &dual_theta_null_square[2];

                    // Compute the last point.
                    let T = &vw12 * &dual_theta_null_square[2];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)

                }

                4 => {

                    // Compute first element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[0];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 += &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_01[1];
                    tmp2 = &uw2 * &dual_theta_null_square[2];
                    tmp1 += &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let X = &tmp3 + &tmp1;

                    // Compute third element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[2];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 += &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_01[0];
                    tmp2 = &uw2 * &dual_theta_null_square[0];
                    tmp1 += &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Z = &tmp3 + &tmp1;

                    // Compute third element
                    tmp1 = &uuvv_m_ww2 * &uuvv_m_ww1;
                    let Y = &tmp1 * &dual_theta_null_square[1];

                    // Compute the last point.
                    let T = &vw12 * &dual_theta_null_square[1];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)

                }

                8 => {

                    // Compute second element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[1];
                    tmp2 = &uw2 * &element_01[1];
                    tmp1 += &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_01[0];
                    tmp2 = &uw2 * &dual_theta_null_square[2];
                    tmp1 += &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Y = &tmp3 + &tmp1;

                    // Compute third element
                    tmp1 = &uuvv_p_ww2 * &dual_theta_null_square[2];
                    tmp2 = &uw2 * &element_01[0];
                    tmp1 = &tmp1 + &tmp2;
                    tmp3 = &uuvv_p_ww1 * &tmp1;

                    tmp1 = &uuvv_p_ww2 * &element_01[1];
                    tmp2 = &uw2 * &dual_theta_null_square[1];
                    tmp1 = &tmp1 + &tmp2;
                    tmp1 = &uw1 * &tmp1;
                    let Z = &tmp3 + &tmp1;

                    // Compute third element
                    tmp1 = &uuvv_m_ww2 * &uuvv_m_ww1;
                    let X = &tmp1 * &dual_theta_null_square[0];

                    // Compute the last point.
                    let T = &vw12 * &dual_theta_null_square[0];

                    ThetaPoint::from_coords(&X, &Y, &Z, &T)

                }
                _ => {
                    panic!("Position should be between 0 to 8");
                }
            }
        }

        /// Given the two 8 torsion points above the kernel of the gluing isogeny with the precomputed Superglue value,
        ///  compute the codomain theta structure and the two Fq elements needed to compute the image of a point
        /// through the isogeny.
        fn gluing_codomain(
            P1_8: &CouplePoint,
            P2_8: &CouplePoint,
            position: u8,
            sign_flip : bool,
            dual_theta_null_square: [Fq; 4],
            element_01: [Fq; 2],
            element_02: [Fq; 2],
            element_03: [Fq; 2]

        ) -> (ThetaStructure, (Fq, Fq)) {
            // First construct the dual coordinates of the kernel and look
            // for the element which is zero
            // For convenience we pack this as an array instead of a tuple:
            let (P1, P2) = P1_8.points();
            let (Q1, Q2) = P2_8.points();

            let bary_P1 = [P1.get_X(), Fq::ZERO, P1.get_Z()];
            let bary_P2 = [P2.get_X(), Fq::ZERO, P2.get_Z()];
            let bary_Q1 = [Q1.get_X(), Fq::ZERO, Q1.get_Z()];
            let bary_Q2 = [Q2.get_X(), Fq::ZERO, Q2.get_Z()];



            let xAxByCyD: [Fq; 4] = Superglue_Evaluation(bary_P1, bary_P2, position, sign_flip, dual_theta_null_square, element_01,element_02,element_03).list();
            let zAtBzYtD: [Fq; 4] = Superglue_Evaluation(bary_Q1, bary_Q2, position, sign_flip, dual_theta_null_square, element_01,element_02,element_03).list();

            // Codomain coefficients
            let mut ABCD = [Fq::ZERO; 4];
            ABCD[0] = &xAxByCyD[0] * &zAtBzYtD[0];
            ABCD[1] = &xAxByCyD[1] * &zAtBzYtD[0];
            ABCD[2] = &xAxByCyD[0] * &zAtBzYtD[2];

            let alpha_inv =  &xAxByCyD[1] * &zAtBzYtD[2];

            let a = &xAxByCyD[0] * &alpha_inv;
            let b = &xAxByCyD[2] * &ABCD[1];
            
            
            let (A, B, C, D) = to_hadamard(&ABCD[0], &ABCD[1], &ABCD[2], &ABCD[3]);
            let codomain = ThetaStructure::new_from_coords(&A, &B, &C, &D);

            (codomain, (a, b))
        }

        /// Given a point P in E1 x E2, return its image through the gluing isogeny.
        fn gluing_image(
            E1E2: &EllipticProduct,
            P: &CouplePoint,
            T1_8: &CouplePoint,
            a: &Fq,
            b: &Fq,
            position: u8,
            sign_flip : bool,
            dual_theta_null_square: [Fq; 4],
            element_01: [Fq; 2],
            element_02: [Fq; 2],
            element_03: [Fq; 2]

        ) -> ThetaPoint {
            // Find dual coordinates of point to push through

            let (E1, E2) = E1E2.curves();
            let (P1, P2) = P.points();
            let (T1_8_1,T1_8_2) = T1_8.points();

            let bary1 : [Fq; 3];
            let bary2 : [Fq; 3];

            if P1.isinfinity() != 0 {
                bary1 = [T1_8_1.get_X(),Fq::ZERO,T1_8_1.get_Z()];
            } else {
                bary1 = Points_to_barycentric_coordinates(E1, P1, T1_8_1);
            }

            if P2.isinfinity() != 0 {
                bary2 = [T1_8_2.get_X(),Fq::ZERO,T1_8_2.get_Z()];
            } else {
                bary2 = Points_to_barycentric_coordinates(E2, P2, T1_8_2);
            }

            let XaYaZbWb : [Fq; 4]  = Superglue_Evaluation(bary1, bary2, position, sign_flip, dual_theta_null_square, element_01,element_02,element_03).list();

            let xyzt = [ &XaYaZbWb[0] * b, &XaYaZbWb[1] * b, &XaYaZbWb[2] * a, &XaYaZbWb[3] * a];

            let (x, y, z, t) = to_hadamard(&xyzt[0], &xyzt[1], &xyzt[2], &xyzt[3]);

            ThetaPoint::from_coords(&x, &y, &z, &t)
        }



        /// Compute the gluing (2,2)-isogeny from a ThetaStructure computed
        /// from an elliptic product.
        fn gluing_isogeny(
            E1E2: &EllipticProduct,
            P1P2_8: &CouplePoint,
            Q1Q2_8: &CouplePoint,
            image_points: &[CouplePoint],
        ) -> (ThetaStructure, Vec<ThetaPoint>) {
            // First recover the four torsion below the 8 torsion
            let P1P2_4 = E1E2.double(&P1P2_8);
            let Q1Q2_4 = E1E2.double(&Q1Q2_8);

            // Compute the superglue coefficients for these two points
            let (pos, sign_flip, dual_null_square, elem_01, elem_02, elem_03) =
                Compute_Superglue_Coefficients(&P1P2_4, &Q1Q2_4);

            // Now it's time to compute the codomain and image of the isogeny
            // with kernel below T1, and T2.

            let (codomain, (a, b)) = gluing_codomain(&P1P2_8, &Q1Q2_8, pos, sign_flip, dual_null_square, elem_01, elem_02, elem_03);

            // We now want to push through a set of points by evaluating each of them
            // under the action of this isogeny. 
            
            let mut theta_images: Vec<ThetaPoint> = Vec::new();

            for P in image_points.iter() {

                let T_image = gluing_image( E1E2, P, P1P2_8, &a, &b, pos, sign_flip, dual_null_square, elem_01, elem_02, elem_03);

                theta_images.push(T_image);
            }

            (codomain, theta_images)
        }



        // ===================================================================
        // Compting general (2,2)-isogenies between theta structures
        //
        // NOTE: For the two steps before a product structure is reached, we
        // need additional symplectic transforms which is controlled by the
        // `hadamard` array of `bool`s. The purpose of these is to avoid null
        // points (or dual null points) which have zero elements, which are
        // incompatible with the doubling formula.
        // ===================================================================

        /// Given the 8-torsion above the kernel, compute the codomain of the
        /// (2,2)-isogeny and the image of all points in `image_points`
        /// Cost:
        /// Codomain: 8S + 9M
        /// Image: 4S + 4M
        fn two_isogeny(
            T1: &ThetaPoint,
            T2: &ThetaPoint,
            image_points: &mut [ThetaPoint],
            hadamard: [bool; 2],
        ) -> ThetaStructure {
            // Compute the squared theta transform of both elements
            // of the kernel
            let (xA, xB, _, _) = T1.squared_theta();
            let (zA, tB, zC, tD) = T2.squared_theta();

            // Compute the codomain coordinates
            let xAtB = &xA * &tB;
            let zAxB = &zA * &xB;
            let zCtD = &zC * &tD;

            let mut A = &zA * &xAtB;
            let mut B = &tB * &zAxB;
            let mut C = &zC * &xAtB;
            let mut D = &tD * &zAxB;

            // Inverses are precomputed for evaluation below
            let A_inv = &xB * &zCtD;
            let B_inv = &xA * &zCtD;
            let C_inv = D;
            let D_inv = C;

            // Finish computing the codomain coordinates
            // For the penultimate case, we skip the hadamard transformation
            if hadamard[1] {
                (A, B, C, D) = to_hadamard(&A, &B, &C, &D);
            }
            let codomain = ThetaStructure::new_from_coords(&A, &B, &C, &D);

            // Now push through each point through the isogeny
            for P in image_points.iter_mut() {
                let (mut XX, mut YY, mut ZZ, mut TT) = P.coords();
                if hadamard[0] {
                    (XX, YY, ZZ, TT) = to_hadamard(&XX, &YY, &ZZ, &TT);
                    (XX, YY, ZZ, TT) = to_squared_theta(&XX, &YY, &ZZ, &TT);
                } else {
                    (XX, YY, ZZ, TT) = to_squared_theta(&XX, &YY, &ZZ, &TT);
                }

                XX *= &A_inv;
                YY *= &B_inv;
                ZZ *= &C_inv;
                TT *= &D_inv;

                if hadamard[1] {
                    (XX, YY, ZZ, TT) = to_hadamard(&XX, &YY, &ZZ, &TT);
                }

                P.X = XX;
                P.Y = YY;
                P.Z = ZZ;
                P.T = TT;
            }

            codomain
        }

        /// Special function for the case when we are (2,2)-isogenous to a
        /// product of elliptic curves. Essentially the same as above, but with
        /// some small changes to deal with that a dual coordinate is now zero.
        /// Computes the codomain of the (2,2)-isogeny and the image of all
        /// points in `image_points`
        /// Cost:
        /// Codomain: 8S + 13M
        /// Image: 4S + 3M
        fn two_isogeny_to_product(
            T1: &ThetaPoint,
            T2: &ThetaPoint,
            image_points: &mut [ThetaPoint],
        ) -> ThetaStructure {
            // Compute the squared theta transform of both elements
            // of the kernel
            let (mut xA, mut xB, yC, yD) = T1.hadamard();
            (xA, xB, _, _) = to_squared_theta(&xA, &xB, &yC, &yD);

            let (mut zA, mut tB, mut zC, mut tD) = T2.hadamard();
            (zA, tB, zC, tD) = to_squared_theta(&zA, &tB, &zC, &tD);

            // Compute the codomain coordinates
            let zAtB = &zA * &tB;
            let A = &xA * &zAtB;
            let B = &xB * &zAtB;
            let C = &zC * &xA * &tB;
            let D = &tD * &xB * &zA;

            // Inverses are precomputed for evaluation below
            let AB = &A * &B;
            let CD = &C * &D;
            let A_inv = CD * &B;
            let B_inv = CD * &A;
            let C_inv = AB * &D;
            let D_inv = AB * &C;

            let codomain = ThetaStructure::new_from_coords(&A, &B, &C, &D);

            for P in image_points.iter_mut() {
                let (mut XX, mut YY, mut ZZ, mut TT) = P.coords();

                (XX, YY, ZZ, TT) = to_hadamard(&XX, &YY, &ZZ, &TT);
                (XX, YY, ZZ, TT) = to_squared_theta(&XX, &YY, &ZZ, &TT);

                XX *= A_inv;
                YY *= B_inv;
                ZZ *= C_inv;
                TT *= D_inv;

                P.X = XX;
                P.Y = YY;
                P.Z = ZZ;
                P.T = TT;
            }

            codomain
        }

        // ========================================================
        // Compting the symplectic transform to expose the
        // product structure and then compute the correct
        // splitting to Montgomery curves.
        // ========================================================

        // This function is a bit of a mess. Ultimately, we want to know whether
        // given some pair of indices whether we should multiply by minus one.
        // We do this by returning either 0: do nothing, or 0xFF...FF: negate
        // the value, which concretely is performed with set_negcond() on the
        // field element.
        //
        // Mathematically we have a few things to juggle. Firstly, although the
        // index should really be tuples (x, y) for x,y in {0,1} we simply index
        // from {0, ..., 3}. So there is first the identification of:
        //
        // 0 : (0, 0)
        // 1 : (1, 0)
        // 2 : (0, 1)
        // 3 : (1, 1)
        //
        // The next thing we need is the dot product of these indices
        // For example:
        // Take i . j is the dot product, so input (x, y) = (1, 3)
        // corresponds to computing:
        // (1, 0) . (1, 1) = 1*1 + 0*1 = 1
        //
        // This means evaluation of chi means the sign is dictated by
        // => (-1)^(i.j) = (-1)^1 = -1
        //
        // A similar thing is done for all pairs of indices below.
        //
        // TODO: there may be a nicer way to organise this function, but
        // I couldn't find a nice closed form for ±1 from a pair (i, j)
        // which i could compute on the fly without first matching from
        // x,y in {0,..,3} to i,j in {(0,0)...(1,1)} (which would mean
        // using a match anyway!!).
        fn chi_eval(x: &usize, y: &usize) -> u32 {
            match (x, y) {
                (0, 0) => 0,
                (0, 1) => 0,
                (0, 2) => 0,
                (0, 3) => 0,
                (1, 0) => 0,
                (1, 1) => u32::MAX,
                (1, 2) => 0,
                (1, 3) => u32::MAX,
                (2, 0) => 0,
                (2, 1) => 0,
                (2, 2) => u32::MAX,
                (2, 3) => u32::MAX,
                (3, 0) => 0,
                (3, 1) => u32::MAX,
                (3, 2) => u32::MAX,
                (3, 3) => 0,
                _ => 1,
            }
        }

        /// For a given index (chi, i) compute the level 2,2 constants (square).
        /// The purpose of this is to identify for which (chi, i) this constant
        /// is zero.
        fn level_22_constants_sqr(null_point: &ThetaPoint, chi: &usize, i: &usize) -> Fq {
            let mut U_constant = Fq::ZERO;
            let null_coords = null_point.list();

            for t in 0..4 {
                let mut U_it = &null_coords[t] * &null_coords[i ^ t];
                U_it.set_condneg(chi_eval(chi, &t));
                U_constant += &U_it;
            }
            U_constant
        }

        /// For each possible even index compute the level 2,2 constant. Return
        /// the even index for which this constant is zero. This only fails for
        /// bad input in which case the whole chain would fail. Evaluates all
        /// positions, and so should run in constant time.
        fn identify_even_index(null_point: &ThetaPoint) -> (usize, usize) {
            const EVEN_INDICIES: [(usize, usize); 10] = [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 2),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 3),
            ];
            // Initialise the return tuple
            let mut chi_zero = 0;
            let mut i_zero = 0;

            for (chi, i) in EVEN_INDICIES.iter() {
                let U_sqr = level_22_constants_sqr(null_point, chi, i);

                // When U_sqr is zero, U_sqr_is_zero = 0xFF...FF
                // and 0 otherwise, so we can use this as a mask
                // to select the non-zero index through the loop
                let U_sqr_is_zero = U_sqr.iszero();
                chi_zero |= (*chi as u32 & U_sqr_is_zero);
                i_zero |= (*i as u32 & U_sqr_is_zero);
            }
            (chi_zero as usize, i_zero as usize)
        }

        /// We can precompute 10 different symplectic transforms which
        /// correspond to each of the possible 10 even indicies which could be
        /// zero. We can select the right change of basis by using the above
        /// functions and then selecting the correct map accordingly.
        fn compute_splitting_matrix(null_point: &ThetaPoint) -> [Fq; 16] {
    #[rustfmt::skip]
            const MAPS: [[Fq; 16]; 10] = [
                [
                    Fq::ONE, Fq::ZERO, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ONE, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ZERO, Fq::ZERO, Fq::ONE,
                    Fq::ZERO, Fq::ZERO, Fq::MINUS_ONE, Fq::ZERO,
                ],
                [
                    Fq::ONE, Fq::ZERO, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ONE, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ZERO, Fq::ONE, Fq::ZERO,
                    Fq::ZERO, Fq::ZERO, Fq::ZERO, Fq::ONE,
                ],
                [
                    Fq::ONE, Fq::ZERO, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ONE, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ZERO, Fq::ONE, Fq::ZERO,
                    Fq::ZERO, Fq::ZERO, Fq::ZERO, Fq::MINUS_ONE,
                ],
                [
                    Fq::ONE, Fq::ONE, Fq::ONE, Fq::ONE,
                    Fq::ONE, Fq::MINUS_ONE, Fq::ONE, Fq::MINUS_ONE,
                    Fq::ONE, Fq::MINUS_ONE, Fq::MINUS_ONE, Fq::ONE,
                    Fq::ONE, Fq::ONE, Fq::MINUS_ONE, Fq::MINUS_ONE,
                ],
                [
                    Fq::ONE, Fq::ZERO, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ZERO, Fq::ZERO, Fq::ONE,
                    Fq::ZERO, Fq::ZERO, Fq::ONE, Fq::ZERO,
                    Fq::ZERO, Fq::MINUS_ONE, Fq::ZERO, Fq::ZERO,
                ],
                [
                    Fq::ONE, Fq::ZERO, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ONE, Fq::ZERO, Fq::ZERO,
                    Fq::ZERO, Fq::ZERO, Fq::ZERO, Fq::ONE,
                    Fq::ZERO, Fq::ZERO, Fq::ONE, Fq::ZERO,
                ],
                [
                    Fq::ONE, Fq::ONE, Fq::ONE, Fq::ONE, Fq::ONE,
                    Fq::MINUS_ONE, Fq::ONE, Fq::MINUS_ONE,
                    Fq::ONE, Fq::MINUS_ONE, Fq::MINUS_ONE, Fq::ONE,
                    Fq::MINUS_ONE, Fq::MINUS_ONE, Fq::ONE, Fq::ONE,
                ],
                [
                    Fq::ONE, Fq::ONE, Fq::ONE, Fq::ONE,
                    Fq::ONE, Fq::MINUS_ONE, Fq::ONE, Fq::MINUS_ONE,
                    Fq::ONE, Fq::ONE, Fq::MINUS_ONE, Fq::MINUS_ONE,
                    Fq::MINUS_ONE, Fq::ONE, Fq::ONE, Fq::MINUS_ONE,
                ],
                [
                    Fq::ONE, Fq::ONE, Fq::ONE, Fq::ONE, Fq::ONE,
                    Fq::MINUS_ONE, Fq::MINUS_ONE, Fq::ONE,
                    Fq::ONE, Fq::ONE, Fq::MINUS_ONE, Fq::MINUS_ONE,
                    Fq::MINUS_ONE, Fq::ONE, Fq::MINUS_ONE, Fq::ONE,
                ],
                [
                    Fq::ONE, Fq::ZETA, Fq::ONE, Fq::ZETA,
                    Fq::ONE, Fq::MINUS_ZETA, Fq::ONE, Fq::ZETA,
                    Fq::ONE, Fq::ZETA, Fq::ONE, Fq::MINUS_ZETA,
                    Fq::ONE, Fq::ZETA, Fq::ONE, Fq::ZETA,
                ],
            ];

            // Identity the current location of the zero
            let zero_location = identify_even_index(null_point);

            // Compute the corresponding matrix to map the zero to
            // the desired place
            // TODO: is a match like this the best thing to do in Rust??
            let M: [Fq; 16];
            match zero_location {
                (0, 2) => M = MAPS[0],
                (3, 3) => M = MAPS[1],
                (0, 3) => M = MAPS[2],
                (2, 1) => M = MAPS[3],
                (0, 1) => M = MAPS[4],
                (1, 2) => M = MAPS[5],
                (2, 0) => M = MAPS[6],
                (3, 0) => M = MAPS[7],
                (1, 0) => M = MAPS[8],
                (0, 0) => M = MAPS[9],
                // The above locations are an exhaustive list of possible inputs, not sure how to tell rust this...
                _ => panic!("Unreachable"),
            }

            M
        }

        /// Map from a theta point to one which admits a splitting to elliptic
        /// products. Essentially requires computing the correct splitting
        /// matrix and then applying the isomorphism
        fn splitting_isomorphism(
            Th: ThetaStructure,
            image_points: &mut [ThetaPoint],
        ) -> ThetaStructure {
            // Compute the correct splitting matrix
            let mut O0 = Th.null_point();
            let M = compute_splitting_matrix(&O0);

            // Map the Theta Structure through the symplectic transform
            apply_base_change(&mut O0, M);

            // Map the points through the symplectic transform
            for P in image_points.iter_mut() {
                apply_base_change(P, M);
            }

            ThetaStructure::new_from_point(&mut O0)
        }

        /// Given a Theta point in the correct representation, compute two
        /// dimension 1 theta points.
        /// Algorithm from:
        /// Models of Kummer lines and Galois representation,
        /// Razvan Barbulescu, Damien Robert and Nicolas Sarkis
        fn split_theta_point(P: &ThetaPoint) -> ((Fq, Fq), (Fq, Fq)) {
            let (a, b, _, d) = P.coords();

            let P1 = (a, b);
            let P2 = (b, d);

            (P1, P2)
        }

        /// Given a dimension one null theta point, compute the corresponding
        /// elliptic curve in the Montgomery model by recovering the Montgomery
        /// coefficient A
        /// Algorithm from:
        /// Models of Kummer lines and Galois representation,
        /// Razvan Barbulescu, Damien Robert and Nicolas Sarkis
        fn null_point_to_montgomery_curve(O0: &(Fq, Fq)) -> Curve {
            let (a, b) = O0;

            let aa = a.square();
            let bb = b.square();

            let T1 = &aa + &bb;
            let T2 = &aa - &bb;

            let A = -&(&T1.square() + &T2.square()) / &(&T1 * &T2);

            Curve::new(&A)
        }

        /// Given a dimension one theta point, compute the corresponding
        /// elliptic curve point on the Kummer line (X : Z)
        /// Algorithm from:
        /// Models of Kummer lines and Galois representation,
        /// Razvan Barbulescu, Damien Robert and Nicolas Sarkis
        fn theta_point_to_montgomery_point(O0: &(Fq, Fq), P: &(Fq, Fq)) -> PointX {
            let (a, b) = O0;
            let (U, V) = P;

            let X = a * V + b * U;
            let Z = a * V - b * U;

            // TODO: rather than use this Type we could directly lift here instead...
            // exposing PointX to be public was the easiest way to keep everything from
            // eccore the same, which will help in the future
            PointX::new_xz(&X, &Z)
        }

        /// Given a ThetaStructure and set of points in the compatible form,
        /// compute the product of elliptic curves and affine points on these
        /// curves.
        fn split_to_product(
            Th: &ThetaStructure,
            image_points: &[ThetaPoint],
            num_image_points: usize,
        ) -> (EllipticProduct, Vec<CouplePoint>) {
            // First we take the domain theta null point and
            // split this to two level-1 theta null points
            let null_point = Th.null_point();
            let (O1, O2) = split_theta_point(&null_point);

            // Compute Montgomery curve from dimension one
            // null points
            let E3 = null_point_to_montgomery_curve(&O1);
            let E4 = null_point_to_montgomery_curve(&O2);
            let E3E4 = EllipticProduct::new(&E3, &E4);

            // Now compute points on E3 x E4
            let mut C: CouplePoint;
            let mut couple_points: Vec<CouplePoint> = vec![];

            for P in image_points.iter().take(num_image_points) {
                // Split to level 1
                let (P1, P2) = split_theta_point(P);
                // Compute the XPoint (X : Z) from each theta point
                let Q1X = theta_point_to_montgomery_point(&O1, &P1);
                let Q2X = theta_point_to_montgomery_point(&O2, &P2);

                // Lift these points to (X : Y : Z) on the curves
                let (Q1, _) = E3.complete_pointX(&Q1X);
                let (Q2, _) = E4.complete_pointX(&Q2X);

                // Package these points into a CouplePoint on
                // E3 x E4
                C = CouplePoint::new(&Q1, &Q2);

                // Push this into the output
                couple_points.push(C);
            }

            (E3E4, couple_points)
        }

        // ========================================================
        // Main Method! Compute the isogeny between elliptic
        // products
        // ========================================================

        // A general comment about "optimal strategies" -- For the isogeny we
        // have a kernel of order 2^n, and to make a step on the (2,2)-isogeny
        // graph what we want to do is scale this point to get 2^(n-1) * P,
        // which is a point of order two, which then allows us to compute the
        // codomain and images. However, as doubling is more expensive than
        // computing an image (in the general case) it's worth storing many
        // values along the way while doubling and pushing them all through the
        // isogeny. As these pushed through points have the same order, the
        // subsequent steps will need to be doubled less (at the cost of needing
        // to push through more points.)
        //
        // For a proper reference, see Sec 4.2 of
        // https://eprint.iacr.org/2011/506.pdf
        //
        // Gluing: Doubling cost: 16M 16S (We have to double two elliptic curve
        // points) Image cost: 76M + 18S + 1I (Very expensive!)
        //
        // All other steps: Doubling cost: 8S + 6M Image cost: 4S + 3M
        //
        // So ignoring the gluing step, we see images have 1/2 the cost
        // (mathematically this is expected as our doubling formula is
        // essentially just two isogeny images) and so the optimised strategy
        // is computed with a weight that doubling is 2X images.
        //
        // For a function to see how strategies are computed, see strategy.py
        // The current implementation "knows" that the gluing is more expensive
        // and so has extra costs for the leftmost branch of the tree.

        /// Compute an isogeny between elliptic products, naive method with no
        /// optimised strategy. Only here for benchmarking
        pub fn product_isogeny_no_strategy(
            E1E2: &EllipticProduct,
            P1P2: &CouplePoint,
            Q1Q2: &CouplePoint,
            image_points: &[CouplePoint],
            n: usize,
        ) -> (EllipticProduct, Vec<CouplePoint>) {
            // Store the number of image points we wish to evaluate to
            // ensure we return them all from the points we push through
            let num_image_points = image_points.len();

            // Convert the &[...] to a vector so we can add points to this
            // dynamically during the optimal strategy
            let mut kernel_couple_pts = image_points.to_vec();

            // Include the kernel inside the vector of points
            // to evaluate. At each step, every element of the
            // vector should be evaluated
            kernel_couple_pts.push(*P1P2);
            kernel_couple_pts.push(*Q1Q2);

            // Compute points of order 8
            let P1P2_8 = E1E2.double_iter(&P1P2, n - 1);
            let Q1Q2_8 = E1E2.double_iter(&Q1Q2, n - 1);

            // Compute Gluing isogeny

            
            let (mut domain, mut kernel_pts) =
                gluing_isogeny(&E1E2, &P1P2_8, &Q1Q2_8, &kernel_couple_pts);


            // Do all remaining steps
            let mut Tp1: ThetaPoint;
            let mut Tp2: ThetaPoint;
            for k in 1..n {
                // Repeatedly double to obtain points in the 8-torsion below the kernel
                Tp1 = domain.double_iter(&kernel_pts[num_image_points], n - k - 1);
                Tp2 = domain.double_iter(&kernel_pts[num_image_points + 1], n - k - 1);

                // For the last two steps, we need to be careful because of the zero-null
                // coordinates appearing from the product structure. To avoid these, we
                // use the hadamard transform to avoid them,
                if k == (n - 2) {
                    domain = two_isogeny(&Tp1, &Tp2, &mut kernel_pts, [false, false])
                } else if k == (n - 1) {
                    domain = two_isogeny_to_product(&Tp1, &Tp2, &mut kernel_pts)
                } else {
                    domain = two_isogeny(&Tp1, &Tp2, &mut kernel_pts, [false, true])
                }
            }

            // Use a symplectic transform to first get the domain into a compatible form
            // for splitting
            domain = splitting_isomorphism(domain, &mut kernel_pts);

            // Split from the level 2 theta model to the elliptic product E3 x E4 and map points
            // onto this product
            let (product, couple_points) = split_to_product(&domain, &kernel_pts, num_image_points);

            (product, couple_points)
        }

        /// Compute an isogeny between elliptic products, use an optimised
        /// strategy for all steps assuming doubling is always more expensive
        /// that images, which is not true for gluing.
        pub fn product_isogeny(
            E1E2: &EllipticProduct,
            P1P2: &CouplePoint,
            Q1Q2: &CouplePoint,
            image_points: &[CouplePoint],
            n: usize,
            strategy: &[usize],
        ) -> (EllipticProduct, Vec<CouplePoint>) {
            // Store the number of image points we wish to evaluate to
            // ensure we return them all from the points we push through
            let num_image_points = image_points.len();

            // Convert the &[...] to a vector so we can add points to this
            // dynamically during the optimal strategy
            let mut kernel_couple_pts = image_points.to_vec();

            // Include the kernel inside the vector of points
            // to evaluate. At each step, every element of the
            // vector should be evaluated
            kernel_couple_pts.push(*P1P2);
            kernel_couple_pts.push(*Q1Q2);

            // Bookkeeping for optimised strategy
            let mut strat_idx = 0;
            let mut level: Vec<usize> = vec![0];
            let mut prev: usize = level.iter().sum();

            // =======================================================
            // Gluing Step
            // TODO:
            // Because of type differences there's annoying code reuse
            // for the optimal strategy here and again for every step
            // in the chain thereafter. Which is bothersome. Maybe there
            // is a better way to write this...
            // =======================================================
            let mut ker1 = *P1P2;
            let mut ker2 = *Q1Q2;

            while prev != (n - 1) {
                // Add the next strategy to the level
                level.push(strategy[strat_idx]);

                // Double the points according to the strategy
                ker1 = E1E2.double_iter(&ker1, strategy[strat_idx]);
                ker2 = E1E2.double_iter(&ker2, strategy[strat_idx]);

                // Add these points to the image points
                kernel_couple_pts.push(ker1);
                kernel_couple_pts.push(ker2);

                // Update the strategy bookkeepping
                prev += strategy[strat_idx];
                strat_idx += 1;
            }

            // Clear out the used kernel point and update level
            kernel_couple_pts.pop();
            kernel_couple_pts.pop();
            level.pop();

            // Compute Gluing isogeny
            let (mut domain, mut kernel_pts) =
                gluing_isogeny(&E1E2, &ker1, &ker2, &kernel_couple_pts);

            // ======================================================
            // All other steps
            // Compute the (2^n-1, 2^n-1)-chain in the theta model
            // =======================================================

            let mut Tp1: ThetaPoint;
            let mut Tp2: ThetaPoint;
            let mut kernel_len: usize;

            // Do all remaining steps
            for k in 1..n {
                prev = level.iter().sum();
                kernel_len = kernel_pts.len();

                Tp1 = kernel_pts[kernel_len - 2];
                Tp2 = kernel_pts[kernel_len - 1];

                while prev != (n - 1 - k) {
                    // Add the next strategy to the level
                    level.push(strategy[strat_idx]);

                    // Double the points according to the strategy
                    Tp1 = domain.double_iter(&Tp1, strategy[strat_idx]);
                    Tp2 = domain.double_iter(&Tp2, strategy[strat_idx]);

                    // Add these points to the image points
                    kernel_pts.push(Tp1);
                    kernel_pts.push(Tp2);

                    // Update the strategy bookkeepping
                    prev += strategy[strat_idx];
                    strat_idx += 1;
                }

                // Clear out the used kernel point and update level
                kernel_pts.pop();
                kernel_pts.pop();
                level.pop();

                // For the last two steps, we need to be careful because of the zero-null
                // coordinates appearing from the product structure. To avoid these, we
                // use the hadamard transform to avoid them,
                if k == (n - 2) {
                    domain = two_isogeny(&Tp1, &Tp2, &mut kernel_pts, [false, false])
                } else if k == (n - 1) {
                    domain = two_isogeny_to_product(&Tp1, &Tp2, &mut kernel_pts)
                } else {
                    domain = two_isogeny(&Tp1, &Tp2, &mut kernel_pts, [false, true])
                }
            }

            // Use a symplectic transform to first get the domain into a compatible form
            // for splitting
            domain = splitting_isomorphism(domain, &mut kernel_pts);

            // Split from the level 2 theta model to the elliptic product E3 x E4 and map points
            // onto this product
            let (product, couple_points) = split_to_product(&domain, &kernel_pts, num_image_points);

            (product, couple_points)
        }
    };
} // End of macro: define_theta_structure

pub(crate) use define_theta_structure;
