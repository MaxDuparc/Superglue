from sage.all import Matrix, ZZ

from theta_structures.couple_point import CouplePoint
from theta_structures.dimension_two import ThetaStructure, ThetaPoint
from theta_isogenies.isogeny import ThetaIsogeny
from utilities.batched_inversion import batched_inversion


def barycentric_coordinates(P,Q,A):
    """
    Compute barycentric coordinates of P ± Q on a Montgomery curve E_A.

    Parameters:
        P, Q : Points in projective coordinates (x:y:z)
        A    : Curve parameter from the Montgomery curve

    Returns:
        (u, v, w) : Barycentric coordinates
    """
    
    xP, yP, zP = P
    xQ, yQ, zQ = Q

    t1 = zP * zQ
    t2 = xP * zQ
    t3 = zP * xQ
    t4 = yP * zQ
    t5 = zP * yQ
    t6 = t4 * t5
    t6 = t6 * t1
    v  = 2 * t6

    t7 = A * t1
    t2_plus_t3 = t2 + t3
    t7 = t7 + t2_plus_t3

    t3_doubled = 2 * t3
    t2_diff = t2_plus_t3 - t3_doubled
    t2_squared = t2_diff**2

    t7 = t7 * t2_squared

    t4 = t4**2
    t5 = t5**2
    t4 = t4 + t5
    t4 = t4 * t1
    u  = t4 - t7

    w = t2_squared * t1

    return u, v, w



class GluingThetaIsogeny(ThetaIsogeny):
    """
    Compute the gluing isogeny from E1 x E2 (Elliptic Product) -> A (Theta Model)

    Expected input:

    - (K1_8, K2_8) The 8-torsion above the kernel generating the isogeny
    - M (Optional) a base change matrix, if this is not including, it can
      be derived from [2](K1_8, K2_8)
    """

    def __init__(self, K1_8, K2_8):
        # Double points to get four-torsion, we always need one of these, used
        # for the image computations but we'll need both if we wish to derived
        # the base change matrix as well
                
        K1_4 = K1_8.double()

        # If M is not included, compute the matrix on the fly from the four
        # torsion.
        K2_4 = K2_8.double()
        g00, g01, g02, g03, position = self.get_base_change_matrix(K1_4, K2_4)

        self.position = position
        self.g00 = g00
        self.g01 = g01
        self.g02 = g02
        self.g03 = g03
                
        # Initalise self
        self.T_shift = K1_8
        self.precom_image_K1_8 = None
        self._precomputation = None
        self._zero_idx = 3
        
        # Compute the codomain of the gluing isogeny
        self._codomain = self._special_compute_codomain(K1_8, K2_8)

    @staticmethod
    def get_base_change_matrix(T1, T2):
        """
        Given the four torsion above the kernel generating the gluing isogeny,
        compute the matrix M which allows us to map points on an elliptic
        product to the compatible theta structure.
        """



        def get_symmetric_element(P1,P2, zeta):
            """
            Compute the symetric ellements of all elements in 

            NOTE: Assumes that Z = 1 
            """
            x1 = P1.x()
            x2 = P2.x()
            z1 = 1
            z2 = 1
            # Determine the case
            if x1 == z1:
                case = 0
            elif x1 == -z1:
                case = 1
            elif x2 == z2:
                case = 2
            elif x2 == -z2:
                case = 3
            elif zeta*(x1+z1)*(x2-z2) == -(x1-z1)*(x2+z2):
                case = 4
            elif zeta*(x1+z1)*(x2-z2) == (x1-z1)*(x2+z2):
                case = 5
            else:
                raise ValueError("No matching case found.")
            # Now treat each case
            if case == 0:
                λ = x2**2 - z2**2
                gT1 = Matrix([[0, λ], [λ, 0]])
                gT2 = Matrix([[x2**2 + z2**2, -2*x2*z2],
                            [2*x2*z2, -(x2**2 + z2**2)]])
                gT12 = Matrix([[zeta*2*x2*z2, -zeta*(x2**2 + z2**2)],
                            [zeta*(x2**2 + z2**2), -zeta*2*x2*z2]])
            elif case == 1:
                λ = x2**2 - z2**2
                gT1 = Matrix([[0, -λ], [-λ, 0]])
                gT2 = Matrix([[x2**2 + z2**2, -2*x2*z2],
                            [2*x2*z2, -(x2**2 + z2**2)]])
                gT12 = Matrix([[-zeta*2*x2*z2, zeta*(x2**2 + z2**2)],
                            [-zeta*(x2**2 + z2**2), zeta*2*x2*z2]])
            elif case == 2:
                λ = x1**2 - z1**2
                gT1 = Matrix([[x1**2 + z1**2, -2*x1*z1],
                            [2*x1*z1, -(x1**2 + z1**2)]])
                gT2 = Matrix([[0, λ], [λ, 0]])
                gT12 = Matrix([[-zeta*2*x1*z1, zeta*(x1**2 + z1**2)],
                            [-zeta*(x1**2 + z1**2), zeta*2*x1*z1]])
            elif case == 3:
                λ = x1**2 - z1**2
                gT1 = Matrix([[x1**2 + z1**2, -2*x1*z1],
                            [2*x1*z1, -(x1**2 + z1**2)]])
                gT2 = Matrix([[0, -λ], [-λ, 0]])
                gT12 = Matrix([[zeta*2*x1*z1, -zeta*(x1**2 + z1**2)],
                            [zeta*(x1**2 + z1**2), -zeta*2*x1*z1]])
            elif case == 4:
                λ = x1**2 - z1**2
                gT1 = Matrix([[x1**2 + z1**2, -2*x1*z1],
                            [2*x1*z1, -(x1**2 + z1**2)]])
                gT2 = Matrix([[-zeta*2*x1*z1, zeta*(x1**2 + z1**2)],
                            [-zeta*(x1**2 + z1**2), zeta*2*x1*z1]])
                gT12 = Matrix([[0, -λ], [-λ, 0]])
            elif case == 5:
                λ = x1**2 - z1**2
                gT1 = Matrix([[x1**2 + z1**2, -2*x1*z1],
                            [2*x1*z1, -(x1**2 + z1**2)]])
                gT2 = Matrix([[zeta*2*x1*z1, -zeta*(x1**2 + z1**2)],
                            [zeta*(x1**2 + z1**2), -zeta*2*x1*z1]])
                gT12 = Matrix([[0, λ], [λ, 0]])
            
            return gT1, gT2, gT12, λ, case          
                    
        
        # Extract elliptic curve points from CouplePoints
        P1, P2 = T1.points()
        Q1, Q2 = T2.points()
        zeta = P1.curve().base_field().gen()

        # Compute symetric element from points from points
        
        G1, G2, G3, lamd1, pos1 = get_symmetric_element(P1,Q1,zeta)
        H1, H2, H3, lamd2, pos2 = get_symmetric_element(P2,Q2,zeta)
        posi = 3*(pos1>>1) + (pos2>>1)
        
        sgl1 = (posi == 0)
        sgl2 = (posi == 4)
        sgl3 = (posi == 8)
        nsgl = not(sgl1 or sgl2 or sgl3)
                                
        lambdda = lamd1*lamd2
        
        # start the trace with id
        a = lambdda
        b = 0
        c = 0
        d = 0

        # T1
        a += G1[0,0] * H1[0,0]
        b += G1[0,0] * H1[0,1]
        c += G1[0,1] * H1[0,0]
        d += G1[0,1] * H1[0,1]

        # T2
        a += G2[0,0] * H2[0,0]
        b += G2[0,0] * H2[0,1]
        c += G2[0,1] * H2[0,0]
        d += G2[0,1] * H2[0,1]


        # T1+T2
        a -= G3[0,0] * H3[0,0]
        b -= G3[0,0] * H3[0,1]
        c -= G3[0,1] * H3[0,0]
        d -= G3[0,1] * H3[0,1]
        
        # Now we act by (0, Q2)
        a1 = H2[0,0] * a + H2[1,0] * b
        if(sgl1 or sgl3):
            z = lamd1*(H2[0,1] * a + H2[1,1] * b)

        a2 = G1[0,0] * a + G1[1,0] * c
        if(sgl2):
            z =  lamd2*(G1[0,0] * b + G1[1,0] * d)

        a *= lambdda 
        b *= lambdda 
        c *= lambdda 
        d *= lambdda 
        
        a1 *=lamd1
        a2 *=lamd2
        
        g00 = [a1**2 + a2**2,a**2 - a1**2 ,a**2 - a2**2 ,0]
        if(sgl1):
            g01 = [a*b + a1*z, a*b - a1*z]
            sign_flip = not(b == c)
            if sign_flip:
                g02 = [-g01[1],-g01[0]]
                g03 = [-g00[1],-g00[0],-g00[3],-g00[2]]
            else:
                g02 = [g01[1],g01[0]]
                g03 = [g00[1],g00[0],g00[3],g00[2]]

            
        if(sgl2):
            g01 = [a*b + a2*z, a*b - a2*z]
            
            sign_flip = not(b == c)
            if sign_flip:
                g02 = [-g01[1],-g01[0]]
                g03 = [-g00[2],-g00[3],-g00[0],-g00[1]]
            else:
                g02 = [g01[1],g01[0]]
                g03 = [g00[2],g00[3],g00[0],g00[1]]
            
        if(sgl3):
            g01 = [a*b + a1*z, a*b - a1*z]
            sign_flip = not(b == c)
            if sign_flip:
                g02 = [-g01[1],-g01[0]]
                g03 = [-g00[3],-g00[2],-g00[1],-g00[0]]
            else:
                g02 = [g01[1],g01[0]]
                g03 = [g00[3],g00[2],g00[1],g00[0]]

        if(nsgl):
            
            g01 = [a*b + c*d, a*b - c*d]
            g02 = [a*c + b*d, a*c - b*d]
            g03 = [a*d + b*c, a*d - b*c]
        
        return g00, g01, g02, g03, posi



    def _superglue_evaluation(self, add_comp1, add_comp2):  
        # Precomputations
        uuvv_p_ww1 = (add_comp1[0] + add_comp1[1]) * (add_comp1[0] - add_comp1[1])  # u1^2 - v1^2
        uuvv_p_ww2 = (add_comp2[0] + add_comp2[1]) * (add_comp2[0] - add_comp2[1])  # u2^2 - v2^2

        tmp1 = add_comp1[2]**2  # w1^2
        tmp2 = add_comp2[2]**2  # w2^2

        uuvv_m_ww1 = uuvv_p_ww1 - tmp1
        uuvv_p_ww1 += tmp1
        uuvv_m_ww2 = uuvv_p_ww2 - tmp2
        uuvv_p_ww2 += tmp2

        uw1 = 2 * add_comp1[0] * add_comp1[2]
        uw2 = 2 * add_comp2[0] * add_comp2[2]

        vw12 = 4 * add_comp1[1] * add_comp2[1] * add_comp1[2] * add_comp2[2]

        match self.position: 
            
            case 1:
                tmp1 = uuvv_p_ww2 * self.g00[0]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                x = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                x += tmp1

                tmp1 = uuvv_p_ww1 * self.g00[1]
                tmp2 = uw1 * self.g02[1]
                tmp1 += tmp2
                y = uuvv_m_ww2 * tmp1

                tmp1 = uuvv_p_ww2 * self.g00[2]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                z = uuvv_m_ww1 * tmp1

                t = vw12 * self.g03[1]

                return x,y,z,t
            
            case 2:
                tmp1 = uuvv_p_ww2 * self.g00[1]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                y = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                y += tmp1

                tmp1 = uuvv_p_ww1 * self.g00[0]
                tmp2 = uw1 * self.g02[1]
                tmp1 += tmp2
                x = uuvv_m_ww2 * tmp1

                tmp1 = uuvv_p_ww2 * self.g00[2]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                z = uuvv_m_ww1 * tmp1

                t = vw12 * self.g03[1]

                return x,y,z,t

            case 3:
                tmp1 = uuvv_p_ww2 * self.g00[0]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                x = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                x += tmp1

                tmp1 = uuvv_p_ww1 * self.g00[2]
                tmp2 = uw1 * self.g02[1]
                tmp1 += tmp2
                z = uuvv_m_ww2 * tmp1

                tmp1 = uuvv_p_ww2 * self.g00[1]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                y = uuvv_m_ww1 * tmp1

                t = vw12 * self.g03[1]

                return x,y,z,t
            
            case 5:
                tmp1 = uuvv_p_ww2 * self.g00[2]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                z = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                z += tmp1

                tmp1 = uuvv_p_ww1 * self.g00[0]
                tmp2 = uw1 * self.g02[1]
                tmp1 += tmp2
                x = uuvv_m_ww2 * tmp1

                tmp1 = uuvv_p_ww2 * self.g00[1]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                y = uuvv_m_ww1 * tmp1

                t = vw12 * self.g03[1]

                return x,y,z,t
            
            case 6:
                tmp1 = uuvv_p_ww2 * self.g00[1]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                y = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                y += tmp1

                tmp1 = uuvv_p_ww1 * self.g00[2]
                tmp2 = uw1 * self.g02[1]
                tmp1 += tmp2
                z = uuvv_m_ww2 * tmp1

                tmp1 = uuvv_p_ww2 * self.g00[0]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                x = uuvv_m_ww1 * tmp1

                t = vw12 * self.g03[1]

                return x,y,z,t

            case 7:
                tmp1 = uuvv_p_ww2 * self.g00[2]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                z = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                z += tmp1

                tmp1 = uuvv_p_ww1 * self.g00[1]
                tmp2 = uw1 * self.g02[1]
                tmp1 += tmp2
                y = uuvv_m_ww2 * tmp1

                tmp1 = uuvv_p_ww2 * self.g00[0]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                x = uuvv_m_ww1 * tmp1

                t = vw12 * self.g03[1]

                return x,y,z,t
            
            case 0:
                tmp1 = uuvv_p_ww2 * self.g00[0]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                x = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                x += tmp1
                

                tmp1 = uuvv_p_ww2 * self.g00[1]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                y = uuvv_p_ww1 * tmp1
                
                tmp1 = uuvv_p_ww2 * self.g02[1]
                tmp2 = uw2 * self.g03[1]
                tmp1 += tmp2
                tmp1 *= uw1
                y += tmp1

                z = uuvv_m_ww1 * uuvv_m_ww2 * self.g00[2]

                t = vw12 * self.g03[3]

                return x,y,z,t
            

            case 4:
                tmp1 = uuvv_p_ww2 * self.g00[0]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                x = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[0]
                tmp1 += tmp2
                tmp1 *= uw1
                x += tmp1
                
                tmp1 = uuvv_p_ww2 * self.g00[2]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                z = uuvv_p_ww1 * tmp1
                
                tmp1 = uuvv_p_ww2 * self.g02[1]
                tmp2 = uw2 * self.g03[2]
                tmp1 += tmp2
                tmp1 *= uw1
                z += tmp1

                y = uuvv_m_ww1 * uuvv_m_ww2 * self.g00[1]

                t = vw12 * self.g03[3]

                return x,y,z,t
            
            case 8:
                tmp1 = uuvv_p_ww2 * self.g00[1]
                tmp2 = uw2 * self.g01[1]
                tmp1 += tmp2
                y = uuvv_p_ww1 * tmp1

                tmp1 = uuvv_p_ww2 * self.g02[1]
                tmp2 = uw2 * self.g03[1]
                tmp1 += tmp2
                tmp1 *= uw1
                y += tmp1
                

                tmp1 = uuvv_p_ww2 * self.g00[2]
                tmp2 = uw2 * self.g01[0]
                tmp1 += tmp2
                z = uuvv_p_ww1 * tmp1
                
                tmp1 = uuvv_p_ww2 * self.g02[0]
                tmp2 = uw2 * self.g03[2]
                tmp1 += tmp2
                tmp1 *= uw1
                z += tmp1

                x = uuvv_m_ww1 * uuvv_m_ww2 * self.g00[0]

                t = vw12 * self.g03[3]

                return x,y,z,t
            
            


    def _special_compute_codomain(self, K1_8, K2_8):
        """
        Given two isotropic points of 8-torsion T1 and T2, compatible with
        the theta null point, compute the level two theta null point A/K_2
        """
        
        xAxByCyD = self._superglue_evaluation((K1_8[0].x(), 0, 1), (K1_8[1].x(), 0, 1))
        zAtBzYtD = self._superglue_evaluation((K2_8[0].x(), 0, 1), (K2_8[1].x(), 0, 1))
        
        # Find the value of the non-zero index
        # Dumb check to make sure everything is OK
        assert xAxByCyD[3] == zAtBzYtD[3] == 0

        # Initialize lists
        # The zero index described the permutation
        ABCD = [xAxByCyD[0]*zAtBzYtD[0],xAxByCyD[1]*zAtBzYtD[0],xAxByCyD[0]*zAtBzYtD[2],0]
        ABCD_inv = [xAxByCyD[1]*zAtBzYtD[2],ABCD[2],ABCD[1],0]
        precom_image_K1_8 = [xAxByCyD[0]*ABCD_inv[0],xAxByCyD[2]*ABCD_inv[2]]
        self.precom_image_K1_8 = precom_image_K1_8
        self._precomputation = ABCD_inv
        
        # Compute non-trivial numerators (Others are either 1 or 0)
        # Final Hadamard of the above coordinates
        a, b, c, d = ThetaPoint.to_hadamard(*ABCD)

        return ThetaStructure([a, b, c, d])

    
    def __call__(self, P):
        """
        Take into input the theta null point of A/K_2, and return the image
        of the point by the isogeny
        """
        if not isinstance(P, CouplePoint):
            raise TypeError(
                "Isogeny image for the gluing isogeny is defined to act on CouplePoints"
            )
            
        # extract X,Z coordinates on pairs of points
        P1, P2 = P.points()
        X1, Z1 = P1[0], P1[2]
        X2, Z2 = P2[0], P2[2]
        
        A1 = P[0].curve().a_invariants()[1]
        A2 = P[1].curve().a_invariants()[1]
        # Correct in the case of (0 : 0)
        if X1 == 0 and Z1 == 0:
            barycenter_coeff1 = (self.T_shift[0][0], 0, self.T_shift[0][2])
        else: 
            barycenter_coeff1 = barycentric_coordinates((X1,P1[1],Z1), self.T_shift[0], A1)

        if X2 == 0 and Z2 == 0:
            barycenter_coeff2 = (self.T_shift[1][0], 0, self.T_shift[1][2])
        else: 
            barycenter_coeff2 = barycentric_coordinates((X2,P2[1],Z2), self.T_shift[1], A2)

        # Compute sum of points on elliptic curve
        xAxByCyD = self._superglue_evaluation(barycenter_coeff1,barycenter_coeff2)
        
        ABCD = [0 for _ in range(4)]
        ABCD[0] = xAxByCyD[0] * self.precom_image_K1_8[1]
        ABCD[1] = xAxByCyD[1] * self.precom_image_K1_8[1]
        ABCD[2] = xAxByCyD[2] * self.precom_image_K1_8[0]
        ABCD[3] = xAxByCyD[3] * self.precom_image_K1_8[0]

        image = ThetaPoint.to_hadamard(*ABCD)
        return self._codomain(image)
