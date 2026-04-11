"""
Bootstrapping CKKS con OpenFHE.

Demuestra cómo el bootstrapping restaura los niveles multiplicativos
de un cifrotexto agotado, permitiendo continuar operando.
Es la solución al «muro del ruido» que vimos con TenSEAL en quick_start.py.

Requisitos: pip install openfhe
"""

import time

from openfhe import (
    CCParamsCKKSRNS,
    FHECKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    SecretKeyDist,
    SecurityLevel,
)


def crear_contexto_bootstrap(niveles_utiles=2, num_slots=4):
    """Crea un contexto CKKS con bootstrapping habilitado.

    Args:
        niveles_utiles: multiplicaciones disponibles entre bootstraps.
        num_slots: número de valores por cifrotexto.

    Returns:
        (cc, keys, profundidad): contexto criptográfico, claves y
        profundidad total configurada.
    """
    params = CCParamsCKKSRNS()

    secret_key_dist = SecretKeyDist.UNIFORM_TERNARY
    params.SetSecretKeyDist(secret_key_dist)
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)

    # El bootstrapping necesita niveles extra para evaluar
    # el circuito de descifrado homomórficamente.
    # level_budget = [niveles_colapsando, niveles_elevando]
    level_budget = [4, 4]
    approx_bootstrap_depth = 8

    profundidad = niveles_utiles + FHECKKSRNS.GetBootstrapDepth(
        approx_bootstrap_depth, level_budget, secret_key_dist
    )

    params.SetMultiplicativeDepth(profundidad)
    params.SetScalingModSize(40)
    params.SetFirstModSize(60)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(PKESchemeFeature.FHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapSetup(level_budget)
    cc.EvalBootstrapKeyGen(keys.secretKey, num_slots)

    return cc, keys, profundidad


def bootstrapping_demo():
    """Replica la Receta 4 (muro del ruido) y lo supera con bootstrapping."""
    cc, keys, profundidad = crear_contexto_bootstrap(
        niveles_utiles=2, num_slots=4
    )

    datos = [3.0, 7.0, 2.0, 5.0]
    ptxt = cc.MakeCKKSPackedPlaintext(datos)
    ctxt = cc.Encrypt(keys.publicKey, ptxt)

    # --- x² (consume 1 nivel) ---
    ctxt = cc.EvalMult(ctxt, ctxt)

    # --- x⁴ (consume otro nivel — en TenSEAL esto fallaba) ---
    ctxt = cc.EvalMult(ctxt, ctxt)

    # Niveles agotados. En Receta 4: «Error: scale out of bounds»
    # Aquí: bootstrapping al rescate.
    t0 = time.perf_counter()
    ctxt = cc.EvalBootstrap(ctxt)
    t_bootstrap = time.perf_counter() - t0

    # --- x⁸ (¡ahora funciona!) ---
    ctxt = cc.EvalMult(ctxt, ctxt)

    # --- Descifrar y verificar ---
    result_ptxt = cc.Decrypt(ctxt, keys.secretKey)
    result_ptxt.SetLength(len(datos))
    valores = [result_ptxt[i] for i in range(len(datos))]

    esperado = [x**8 for x in datos]
    error_max = max(abs(v - e) for v, e in zip(valores, esperado))

    print(f"Resultado FHE: {[f'{v:.1f}' for v in valores]}")
    print(f"Esperado:      {[f'{v:.1f}' for v in esperado]}")
    print(f"Error máximo:  {error_max:.6f}")
    print(f"Tiempo bootstrap: {t_bootstrap:.2f} s")

    return valores, esperado, t_bootstrap


if __name__ == "__main__":
    bootstrapping_demo()
