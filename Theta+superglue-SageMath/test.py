
from sage.all import *
 
#import sage.schemes.curves.projective_curve as plane_curve
# Python imports
import time

# Local imports
from theta_structures.couple_point import CouplePoint
from theta_isogenies.product_isogeny import EllipticProductIsogeny
from theta_isogenies.product_isogeny_sqrt import EllipticProductIsogenySqrt
from montgomery_isogenies.isogenies_x_only import isogeny_from_scalar_x_only, evaluate_isogeny_x_only
from utilities.order import has_order_D
from utilities.discrete_log import BiDLP
from utilities.supersingular import torsion_basis, fix_torsion_basis_renes
from utilities.strategy import optimised_strategy
from utilities.utils import speed_up_sagemath, verbose_print

# Some precomputed
# ea, eb, X, Y
# Such that
# A = 2**ea
# B = 3**eb
# C = A - B = X^2 + Y^2
DIAMONDS = [
    (1, 9, 5, 10, 13),
    (1, 72, 41, 28930079307, 62039858138),
    (105, 126, 75, 4153566748924546849, 8198183225858618766),
    (1, 176, 87, 62060540753699060752145450, 303198714682554770479637193),
    (15, 208, 105, 3624033553412861679508963751522, 19956014635540685051671510447227),
    (
        1,
        602,
        363,
        1150828796773097449350776400976394707767314979521270112577233719378561676605426313207017359,
        3908152402269014765841876930143810585616099277004914795408350071193595839964749987840751014,
    ),
]


def generate_splitting_kernel(param_index=0):
    """ """
    f, ea, eb, X, Y = DIAMONDS[param_index]

    # Configure a prime which allows an easy diamond configuration:
    A = ZZ(2**ea)
    B = ZZ(3**eb)
    p = f * 4 * A * B - 1
    F = GF(p**2, name="i", modulus=[1, 0, 1])

    # Cofactor isogeny is the sum of squares
    C = A - B
    assert C == X**2 + Y**2

    # Starting curve
    E0 = EllipticCurve(F, [1, 0])
    iota = E0.automorphisms()[2]

    # Do Bob's SIDH Key-Exchange
    bob_secret = randint(0, B)

    # Compute torsion basis
    P2, Q2 = torsion_basis(E0, 4 * A)
    P2, Q2 = fix_torsion_basis_renes(P2, Q2, ea + 2)
    P3, Q3 = torsion_basis(E0, B)

    # Check automorphism
    assert iota(iota(P2)) == -P2
    assert iota(iota(Q2)) == -Q2

    phiB, _ = isogeny_from_scalar_x_only(E0, B, bob_secret, basis=(P3, Q3))
    phi_P0, phi_Q0 = evaluate_isogeny_x_only(phiB, P2, Q2, 4 * A, B)

    # We pick values such that the aux. is easy to compute
    def aux_endomorphism(P):
        return X * P + Y * iota(P)

    # Kernel which generates the split isogeny
    P1 = aux_endomorphism(P2)
    Q1 = aux_endomorphism(Q2)
    ker_Phi = (P1, Q1, phi_P0, phi_Q0)

    return ker_Phi, (E0, bob_secret)



def check_result(E0, EA, EB, B, Phi):
    # Push the torsion basis of EB through the (2,2) isogeny
    P3, Q3 = torsion_basis(E0, B)
    PB3, QB3 = torsion_basis(EB, B)

    E3, E4 = Phi.codomain()

    # Find which of the two codomain curves is our starting curve
    if E3.is_isomorphic(E0):
        E_start = E3
        index = 0
    else:
        assert E4.is_isomorphic(E0)
        E_start = E4
        index = 1

    # One of these points will have full order, which we can
    # recover the secret from
    L1 = CouplePoint(EA(0), PB3)
    L2 = CouplePoint(EA(0), QB3)

    t0 = time.process_time()
    K_img = Phi(L1)[index]
    verbose_print(
        f"Computing an image took: {time.process_time() - t0:.5f}", verbose=verbose
    )

    # Ensure we have a point in the full order
    if not has_order_D(K_img, B):
        K_img = Phi(L2)[index]
    assert has_order_D(K_img, B)

    # Isomorphisms back to original curve E0
    isomorphisms = E_start.isomorphisms(E0)
    for iso in isomorphisms:
        K = iso(K_img)

        # Recover secret by solving dlogs
        a, b = BiDLP(K, P3, Q3, B)

        # fix due to the fact we use E1728 as a starting curve
        if gcd(a, B) != 1:
            iota = E0.automorphisms()[2]
            K = iota(K)
            a, b = BiDLP(K, P3, Q3, B)

        # Recover secret from the BiDLP
        secret = (Mod(ZZ(b), B) / a).lift()

        # Ensure the collected secret is correct
        _, EB_test = isogeny_from_scalar_x_only(E0, B, secret, basis=(P3, Q3))

        if EB == EB_test:
            break

    return secret


def test_SIDH_attack(test_index=-1, verbose=False):
    (P1, Q1, P2, Q2), (E0, bob_secret) = generate_splitting_kernel(test_index)

    # Create kernel from CouplePoint data
    ker_Phi = (CouplePoint(P1, P2), CouplePoint(Q1, Q2))
    EA, EB = ker_Phi[0].curves()

    _, ea, eb, _, _ = DIAMONDS[test_index]
    B = ZZ(3**eb)

    strategy = optimised_strategy(ea)
    t0 = time.process_time()
    Phi = EllipticProductIsogeny(ker_Phi, ea, strategy=strategy)
    verbose_print(f"(2,2)-chain took: {time.process_time() - t0:.5f}s", verbose=verbose)

    ker_Phi_scaled = (4 * ker_Phi[0], 4 * ker_Phi[1])
    strategy_sqrt = optimised_strategy(ea - 2)
    t0 = time.process_time()
    Phi2 = EllipticProductIsogenySqrt(ker_Phi_scaled, ea, strategy=strategy_sqrt)
    verbose_print(
        f"(2,2)-sqrt-chain took: {time.process_time() - t0:.5f}s", verbose=verbose
    )

    verbose_print(f"Original secret: {bob_secret}", verbose=verbose)
    verbose_print("Recovering secret via isogeny chain:", verbose=verbose)
    secret = check_result(E0, EA, EB, B, Phi)
    verbose_print(f"Recovered secret: {secret}", verbose=verbose)
    verbose_print("Recovering secret via sqrt isogeny chain:", verbose=verbose)
    secret2 = check_result(E0, EA, EB, B, Phi2)
    verbose_print(f"Recovered secret via sqrt chain: {secret2}", verbose=verbose)

    assert secret == bob_secret, "Secrets do not match!"
    assert secret2 == bob_secret, "Secrets do not match!"


if __name__ == "__main__":
    speed_up_sagemath()
    test_number = 5
    verbose = True

    for test_index in range(len(DIAMONDS)):
        for _ in range(test_number):
            print(f"Testing index: {test_index}")
            test_SIDH_attack(test_index=test_index, verbose=verbose)
            print("")


