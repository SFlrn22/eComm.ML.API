from hashlib import sha256
import base64

input_ = "THISISASUPERSECRETPASSTHATCANNOTBEGUESSED"
print(base64.b64encode(sha256(input_.encode('utf-8')).digest()))