# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed for matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY plotting.py .
COPY colors.txt .
COPY pheno_table.tsv .
COPY snp_gene_pq.txt .
#COPY degas_betas.npy .
#COPY guide_all_100lat_bl_ll.npz .
#COPY w_values.npz .
#COPY betas.npy .
COPY snplist.txt .
COPY pheno_table_enhanced.csv .

#COPY all_phenos/ ./all_phenos/

EXPOSE 8000

CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
