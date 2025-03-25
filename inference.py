import asyncio
import torch
import numpy as np
from nats.aio.client import Client as NATS
from mercurion.model import MercurionMLP
from mercurion.labels import tox21_labels
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import json
from schemas.schemas import InferenceRequest
from pydantic import ValidationError
from jose import jwt, JWTError


# ✔️ Label e indice: definiti una sola volta, in ordine
ALL_LABELS = tox21_labels
TOP4_LABELS = ['SR-ATAD5', 'NR-AhR', 'SR-MMP', 'SR-p53']
TOP4_INDICES = [ALL_LABELS.index(label) for label in TOP4_LABELS]

with open("public.pem", "r") as f:
    PUBLIC_KEY = f.read()

ALGORITHM = "RS256"

def verify_jwt(token: str):
    try:
        payload = jwt.decode(token, PUBLIC_KEY, algorithms=[ALGORITHM])
        return payload  # Puoi anche restituire solo alcune claim se vuoi
    except JWTError as e:
        return None

# ✔️ Preprocessing SMILES → fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol).ToBitString()
    return np.array([int(b) for b in fp], dtype=np.float32)

# ✔️ Inference function
def predict(smiles, model, device):
    fp = smiles_to_fp(smiles)
    if fp is None:
        return {'error': 'Invalid SMILES'}

    tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).cpu().numpy().flatten()
        top4_probs = [float(probs[i]) for i in TOP4_INDICES]
    return dict(zip(TOP4_LABELS, top4_probs))

# ✔️ Main NATS client
async def run():
    nc = NATS()
    await nc.connect("nats://localhost:4223")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MercurionMLP().to(device)
    model.load_state_dict(torch.load('outputs/models/best_model.pt', map_location=device, weights_only=True))
    model.eval()

    async def message_handler(msg):
        try:
            req = InferenceRequest.parse_raw(msg.data.decode())
        except ValidationError as e:
            await msg.respond(json.dumps({"error": f"Invalid request: {e.errors()}"}).encode())
            return

        # 🔐 Validazione token
        user_payload = verify_jwt(req.accessToken)
        if not user_payload:
            await msg.respond(json.dumps({"error": "Invalid or expired access token"}).encode())
            return

        result = predict(req.smiles, model, device)
        await msg.respond(json.dumps(result).encode())



    await nc.subscribe("inference.smiles", cb=message_handler)
    print("✅ Microservizio in ascolto su 'inference.smiles'...")

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run())
