As you may have noticed, concerned citizen, this directory contains a private
RSA key file. This is not an accident, and was done intentionally.

The public key corresponding to the private key in
`private-deploy-key-for-pulling-from-cirq` is registered as a deploy key on the
cirq repository. It allows anyone with access to openfermion-cirq to also read
from cirq.
In particular: tools such as travis-ci.

Everyone with access to openfermion-cirq already has access to cirq. It's okay
for them to have a second way to access cirq. As long as we remember to cycle
the key if we ever cut someone's access...

After all, what could possibly go wrong?...
