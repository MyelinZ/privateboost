Test-only RSA keypairs and certs for JWT unit/integration tests. NEVER use
any of these outside tests.

`test_key.pem` / `test_key.pub.pem`: the dev-issuer keypair the workspace
mints and verifies dev tokens with. Generated with:

    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out test_key.pem
    openssl pkey -in test_key.pem -pubout -out test_key.pub.pem

`jwks_key.pem` / `jwks_key.pub.pem`: a second keypair from the same recipe,
used by the JWT-rejection test (`crates/pbr-server/tests/auth.rs`) as the
"wrong" signer: a token minted with this key under a known `kid` must fail to
verify against `test_key`'s public key. `jwks.json` is a Google-style JWKS
document (kid `jwks-test-1`) whose `n`/`e` were derived from `jwks_key.pub.pem`'s
modulus (`openssl rsa -pubin -in jwks_key.pub.pem -noout -modulus`,
base64url-encoded) and the standard exponent 65537 (`AQAB`); the
`update_keys_from_jwks` tests load it and verify a `jwks_key.pem`-signed
token against it.

`tls/`: a self-signed CA (`ca.crt` / `ca.key`) and one server cert
(`server.crt` / `server.key`) whose only SAN is `IP:127.0.0.1`, used by the
client-facing-TLS tests (`crates/pbr-client/tests/tls.rs`). The client pins
`ca.crt`; rustls matches the endpoint host against the cert's IP SAN.
Regenerate with:

    deploy/hetzner/gen-certs.sh 127.0.0.1 --out crates/pbr-server/tests/fixtures/tls
