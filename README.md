# Scaled Dot-Product Attention (Implementação do Zero)

Este projeto implementa o mecanismo de **Scaled Dot-Product Attention** usando **Python + NumPy**, sem usar camadas prontas de Deep Learning.

## O que é

A attention calcula quais elementos são mais relevantes entre si usando **Q (queries)**, **K (keys)** e **V (values)**.

Fórmula implementada:

\[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

- **Q**: consultas
- **K**: chaves
- **V**: valores
- **\(d_k\)**: dimensão das chaves

> O **softmax é aplicado por linha** (uma distribuição de atenção para cada query).

---

## Arquivos do projeto

- `attention.py` → implementação da attention
- `test_attention.py` → teste com exemplo numérico
- `requirements.txt` → dependências (`numpy`)

---

## Como rodar (passo a passo)

### 1) Entrar na pasta do projeto
```powershell
cd "C:\Users\guilh\Documents\Projetos\lab_p101\lab_01"
```

### 2) Criar o ambiente virtual
```powershell
py -m venv venv
```

> Se `py` não funcionar, tente:
```powershell
python -m venv venv
```

### 3) Ativar o ambiente virtual
**PowerShell**
```powershell
.\venv\Scripts\Activate.ps1
```

**CMD**
```cmd
venv\Scripts\activate.bat
```

> Se der erro de permissão no PowerShell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4) Instalar as dependências
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5) Executar o teste
```powershell
python test_attention.py
```

---

## O que o teste mostra

O arquivo `test_attention.py` imprime:

- `Q`, `K`, `V`
- `scores = Q @ K.T`
- `scaled_scores = scores / sqrt(d_k)`
- `attention_weights = softmax(...)`
- `output final`

Se tudo estiver certo, aparece:

```text
Teste executado com sucesso!
```
