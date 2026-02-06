import csv
import itertools as it
import json
import logging
from pathlib import Path

import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from datetime import date
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import norm

from plotting import var_contrib_to_lat, plot_variance_components, pheno_contribs_to_lat

# general FIXME: use log rather than print

log = logging.getLogger('shiny')
log.setLevel(logging.WARNING)

ALL_LATENT_LABELS = [f'GUIDE Lat{i}' for i in range(100)]
LATENT_LABELS = []
PLOT_HEIGHT = 400


def load_npy(path, mmap_mode='r') -> np.array:
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except ValueError as e:
        log.warning('Cannot memory map numpy array %s, falling back to regular load. (%s)', path, e)
    return np.load(path, allow_pickle=True)


def load_llm_data():
    with open('./GUIDE_browser_llm_prompt.txt', 'r', encoding='utf-8') as f:
        llm_prompt_text = f.read()

    with open('./llm_out.json', 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    llm_latent_choices = {}
    # llm_display_to_key = {}  # Add this reverse mapping  # unused
    for key, factor in llm_data.get('latent_factors', {}).items():
        factor_num = key.replace('Lat', '')
        label = factor.get('label', 'Unknown')
        display_name = f'Latent Factor {factor_num}: {label}'
        llm_latent_choices[key] = display_name
        # llm_display_to_key[display_name] = key

    return llm_prompt_text, llm_data, llm_latent_choices


LLM_PROMPT, LLM_DATA, LLM_LATENT_CHOICES = load_llm_data()


def load_w_values():
    try:
        wvpath = Path('./w_values')
        logw_mat_TL = load_npy(wvpath / 'logw_mat_TL.npy')
        logw_mat_XL = load_npy(wvpath / 'logw_mat_XL.npy')
        log.info('w-values loaded successfully!')
        return True, logw_mat_TL, logw_mat_XL
    except (FileNotFoundError, KeyError):
        log.warning('could not load w-values', exc_info=True)
        return False, None, None


W_VALUES_AVAILABLE, LOGW_MAT_TL, LOGW_MAT_XL = load_w_values()


def load_log10pvalues():
    # there are around 3.8GiB of zscores/pvalues, it's read only so we can
    # memory map it in order to run in a more constrained environment.
    try:
        log10pvalues = load_npy('./log10pvalues.npy')
    except FileNotFoundError:
        log10pvalues = None

    if log10pvalues is None:
        # getting here means we haven't precomputed the log10 pvalues,
        # so we try to do so, and eat the memory usage
        try:
            zscores = load_npy('./zscores.npy', mmap_mode=None)

            # try to use inplace operations to cut down on memory usage
            pvalues = np.abs(zscores, out=zscores)
            for ix, data in enumerate(pvalues):
                pvalues[ix, :] = norm.cdf(data)
            np.subtract(1, pvalues, out=pvalues)
            np.multiply(2, pvalues, out=pvalues)
            np.clip(pvalues, a_min=1e-300, a_max=None, out=pvalues)

            log10pvalues = np.log10(pvalues, out=pvalues)
            np.negative(log10pvalues, out=log10pvalues)
            np.clip(log10pvalues, a_min=0, a_max=None, out=log10pvalues)
        except Exception:
            log.warning('Could not load z-scores', exc_info=True)
            return False, None

    log.info('Z-scores loaded successfully! Shape: %s', log10pvalues.shape)
    return True, log10pvalues


ZSCORES_AVAILABLE, LOG10PVALUES = load_log10pvalues()


def color_pallet(n):
    if n > 12:
        colors_base = plt.cm.tab20c((np.arange(20)).astype(int))
    else:
        colors_base = plt.cm.Paired((np.arange(12)).astype(int))
    colors = [x[1] for x in zip(np.arange(n), it.cycle(colors_base))]
    return colors


def get_safe_phe_name(phe):
    return ''.join([c if (c.isalnum() or c in ['_', '.']) else '_' for c in phe])


def set_latent_labels(ll):
    global LATENT_LABELS
    LATENT_LABELS = ll


def gen_barplot_data(pheno_labels, scores, is_guide, colors):
    method_str = 'GUIDE' if is_guide else 'TSVD'
    ls_dict = dict()
    lat_labels = [f'{method_str} Lat{i}' for i in range(100)]
    for label, row in zip(lat_labels, scores.T):
        ls_dict[label] = row
    df = pd.DataFrame(data=ls_dict, index=pheno_labels)
    df.index.name = 'phenotype'
    try:
        df['color'] = colors
    except:
        pass
    return df


def parse_variant_label(var_label):
    try:
        parts = var_label.split('-')
        chr_val = parts[0]
        if chr_val == 'X':
            chr_val = 23
        elif chr_val == 'Y':
            chr_val = 24
        elif chr_val in ['MT', 'M']:
            chr_val = 25
        else:
            chr_val = int(chr_val)
        pos_val = int(parts[1])
        return chr_val, pos_val
    except:
        return None, None


def prepare_manhattan_data(
    var_labels, contrib_var, trait_name, contrib_phe, var_gene_file='./snp_gene_pq.txt'
):
    chr_list = []
    pos_list = []
    for var in var_labels:
        chr_val, pos_val = parse_variant_label(var)
        chr_list.append(chr_val)
        pos_list.append(pos_val)
    trait_idx = np.where(PHENO_GUIDE == trait_name)[0]
    if len(trait_idx) == 0:
        return None
    trait_idx = trait_idx[0]
    safe_name = get_safe_phe_name(trait_name)
    path = f'./all_phenos/phenotypes/guide/{safe_name}/squared_cosine_scores.tsv'
    try:
        contrib_df = pd.read_csv(path, sep='\t', header=0, index_col=1)
        top_data = contrib_df['squared_cosine_score'].sort_values(ascending=False)[:3]
        top3_latents = [int(idx) for idx in top_data.index]
        top3_latent_labels = [f'GUIDE Lat{idx}' for idx in top3_latents]
        top3_contributions = list(top_data.values)
    except:
        top3_latents_idx = np.argsort(-contrib_phe[trait_idx, :])[:3]
        top3_latents = [int(idx) for idx in top3_latents_idx]
        top3_latent_labels = [f'GUIDE Lat{idx}' for idx in top3_latents]
        top3_contributions = [float(contrib_phe[trait_idx, idx]) for idx in top3_latents]
    if ZSCORES_AVAILABLE and LOG10PVALUES is not None:
        value_data = LOG10PVALUES[:, trait_idx]
        # value_label = 'log10p'  # unused
    else:
        trait_latent_weights = contrib_phe[trait_idx, :]
        value_data = (contrib_var * trait_latent_weights).sum(axis=1)
        # value_label = 'weighted_contrib'  # unused
    df = pd.DataFrame({'CHR': chr_list, 'POS': pos_list, 'VAR': var_labels, 'value': value_data})
    df = df.dropna(subset=['CHR', 'POS'])
    df['CHR'] = df['CHR'].astype(int)
    df['POS'] = df['POS'].astype(int)
    var2gene_dict = {}
    try:
        with open(var_gene_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    var2gene_dict[parts[0]] = parts[1]
    except:
        pass
    df['GENE'] = df['VAR'].map(var2gene_dict).fillna('')
    df['top_latent_rank'] = 0
    for var_idx in range(len(var_labels)):
        variant_loadings_on_top3 = [contrib_var[var_idx, lat_idx] for lat_idx in top3_latents]
        max_loading_idx = np.argmax(variant_loadings_on_top3)
        if variant_loadings_on_top3[max_loading_idx] > 0.01:
            df.loc[var_idx, 'top_latent_rank'] = max_loading_idx + 1
    if W_VALUES_AVAILABLE and LOGW_MAT_XL is not None:
        w_values_for_trait = []
        for var_idx in range(len(var_labels)):
            variant_w_on_top3 = [LOGW_MAT_XL[var_idx, lat_idx] for lat_idx in top3_latents]
            max_w = max(variant_w_on_top3)
            w_values_for_trait.append(max_w)
        df['w_value'] = w_values_for_trait
    else:
        df['w_value'] = np.nan

    if W_VALUES_AVAILABLE and LOGW_MAT_TL is not None:
        top3_w_values = [LOGW_MAT_TL[trait_idx, lat_idx] for lat_idx in top3_latents]
    else:
        top3_w_values = [np.nan, np.nan, np.nan]

    df.attrs['top3_latents'] = top3_latents
    df.attrs['top3_latent_labels'] = top3_latent_labels
    df.attrs['top3_contributions'] = top3_contributions
    df.attrs['top3_w_values'] = top3_w_values
    return df


def plot_manhattan(
    data_df,
    value_col='value',
    title='Manhattan Plots',
    figsize=(16, 6),
    chr_filter=None,
    region_start=None,
    region_end=None,
    use_pvalues=True,
):
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if chr_filter is not None:
        data_df = data_df[data_df['CHR'] == chr_filter].copy()
        if region_start is not None and region_end is not None:
            data_df = data_df[
                (data_df['POS'] >= region_start) & (data_df['POS'] <= region_end)
            ].copy()
    data_df = data_df.sort_values(['CHR', 'POS']).copy()
    if len(data_df) == 0:
        ax.text(
            0.5,
            0.5,
            'No data in selected region',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=14,
        )
        return fig, ax
    log.debug(
        'DEBUG: value column min=%.4f, max=%.4f', data_df[value_col].min(), data_df[value_col].max()
    )
    chromosomes = sorted(data_df['CHR'].unique())
    rank_colors = {1: '#D62728', 2: '#FF7F0E', 3: '#2CA02C', 0: '#A9A9A9'}
    top3_latent_labels = data_df.attrs.get(
        'top3_latent_labels', ['1st Latent', '2nd Latent', '3rd Latent']
    )
    if use_pvalues:
        highlight_threshold = 7.3
    else:
        highlight_threshold = data_df[value_col].quantile(0.95)
    if chr_filter is None:
        data_df['cumpos'] = 0
        cumulative_pos = 0
        chr_centers = {}
        for chrom in chromosomes:
            chr_data = data_df[data_df['CHR'] == chrom]
            chr_len = chr_data['POS'].max() - chr_data['POS'].min()
            data_df.loc[data_df['CHR'] == chrom, 'cumpos'] = (
                chr_data['POS'] - chr_data['POS'].min() + cumulative_pos
            )
            chr_centers[chrom] = cumulative_pos + chr_len / 2
            cumulative_pos += chr_len + 1e6
        x_col = 'cumpos'
    else:
        x_col = 'POS'
        chr_centers = None
    ntotal = len(data_df)
    nsig = (data_df[value_col] > highlight_threshold).sum()
    enrichment_scores = {}
    for rank in [1, 2, 3, 0]:
        mask = data_df['top_latent_rank'] == rank
        gtotal = mask.sum()
        gsig = ((data_df[value_col] > highlight_threshold) & mask).sum()
        if gtotal > 0 and ntotal > 0 and nsig > 0:
            enrichment = (gsig / gtotal) / (nsig / ntotal)
        else:
            enrichment = 1.0
        enrichment_scores[rank] = enrichment
    for rank in [0, 3, 2, 1]:
        mask = data_df['top_latent_rank'] == rank
        if rank > 0 and len(top3_latent_labels) >= rank:
            label = f'{top3_latent_labels[rank - 1]} (En={enrichment_scores[rank]:.2f})'
        else:
            label = f'Other (En={enrichment_scores[rank]:.2f})'
        if mask.any():
            rank_data = data_df[mask]
            ax.scatter(
                rank_data[x_col],
                rank_data[value_col],
                c=rank_colors[rank],
                s=40 if rank > 0 else 20,
                alpha=0.8 if rank > 0 else 0.5,
                edgecolors='none',
                label=label,
                zorder=rank if rank > 0 else 1,
            )
        else:
            ax.scatter(
                [],
                [],
                c=rank_colors[rank],
                s=40 if rank > 0 else 20,
                alpha=0.8 if rank > 0 else 0.5,
                edgecolors='none',
                label=label,
                zorder=rank if rank > 0 else 1,
            )
    if chr_filter is None:
        ax.set_xticks([chr_centers[chr_num] for chr_num in chromosomes])
        ax.set_xticklabels([str(chr_num) for chr_num in chromosomes])
        ax.set_xlabel('Chromosome', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel(f'Position on Chromosome {chr_filter} (bp)', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    if use_pvalues:
        ax.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('Weighted Variance Component', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    y_max = data_df[value_col].max()
    y_min = 0
    y_padding = max(y_max * 0.1, 1.0)
    ax.set_ylim(y_min, y_max + y_padding)
    log.debug('DEBUG: Setting ylim to [%f, %f]', y_min, y_max + y_padding)
    if highlight_threshold <= y_max + y_padding:
        ax.axhline(
            y=highlight_threshold,
            color='red',
            linestyle='--',
            linewidth=1,
            alpha=0.7,
            label=f'Threshold: {highlight_threshold:.2f}',
        )
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 4:
        ordered_handles = handles[3::-1]
        ordered_labels = labels[3::-1]
        if len(handles) > 4:
            ordered_handles.append(handles[4])
            ordered_labels.append(labels[4])
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc='upper left',
            bbox_to_anchor=(1.0, 1),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=9,
        )
    else:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.0, 1),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=9,
        )
    fig.tight_layout(rect=[0.05, 0, 0.85, 1])
    return fig, ax


def compute_gene_contrib_to_cyto(label_var, contribution_var, var_gene_file):
    var2gene: dict[str, str] = {}
    gene2cyto: dict[str, str] = {}
    with open(var_gene_file, 'r') as f:
        records = csv.reader(f, delimiter='\t')
        for var, gene, cyto in records:
            var2gene[var] = gene
            gene2cyto[gene] = cyto
    contribution_var_df = pd.DataFrame(contribution_var)
    contribution_var_df['gene'] = [var2gene.get(x, '') for x in label_var]
    contribution_cyto_df = contribution_var_df.groupby('gene').sum()
    return (
        contribution_cyto_df.to_numpy(),
        np.array(contribution_cyto_df.index),
        gene2cyto,
    )


def gen_data(var_labels, contrib_var_guide, var_gene_file='./snp_gene_pq.txt'):
    cyto_c_guide, names_guide, gene2cyto_dict_guide = compute_gene_contrib_to_cyto(
        var_labels, contrib_var_guide, var_gene_file
    )
    ccg = dict(zip([f'GUIDE Lat{x}' for x in range(100)], cyto_c_guide.T))
    cyto_contrib_guide = pd.DataFrame(data=ccg, index=names_guide)
    return cyto_contrib_guide, gene2cyto_dict_guide


def create_enhanced_phenotype_table(pheno_labels, contrib_phe, w_values_available, logw_mat_TL):
    sorted_indices = np.argsort(-contrib_phe, axis=1)
    rows = []
    for i, pheno in enumerate(pheno_labels):
        latent_info = []
        top3_total = 0.0
        for rank in range(3):
            latent_idx = sorted_indices[i, rank]
            contrib_val = contrib_phe[i, latent_idx]
            top3_total += contrib_val
            if w_values_available and logw_mat_TL is not None:
                w_val = logw_mat_TL[i, latent_idx]
            else:
                w_val = np.nan
            latent_info.append(
                {
                    'rank': rank + 1,
                    'latent': f'Lat{latent_idx}',
                    'latent_idx': int(latent_idx),
                    'contribution': contrib_val,
                    'w_value': w_val,
                }
            )
        rows.append({'Phenotype': pheno, 'latent_info': latent_info, 'top3_total': top3_total})
    df = pd.DataFrame(rows)
    df = df.sort_values('top3_total', ascending=False).reset_index(drop=True)
    return df


GZP = Path('./guide_all_100lat_bl_ll')
CONTRIB_GUIDE = load_npy(GZP / 'contribution_phe.npy')
CONTRIB_VAR_GUIDE = load_npy(GZP / 'contribution_var.npy')
COS_GUIDE = load_npy(GZP / 'cos_phe.npy')
FACTOR_GUIDE = load_npy(GZP / 'factor_phe.npy')
GENE_CONTRIB_GUIDE = load_npy(GZP / 'contribution_gene.npy')
PHENO_GUIDE = load_npy(GZP / 'label_phe.npy')
ALL_PHENOTYPES = list(PHENO_GUIDE)
VAR_LABELS = load_npy(GZP / 'label_var.npy')
del GZP

COLORS = list(pd.read_csv('./colors.txt', sep='\t', header=None)[0])

CYTO_CONTRIB_GUIDE, GENE2CYTO_DICT_GUIDE = gen_data(VAR_LABELS, CONTRIB_VAR_GUIDE)
ENHANCED_PHENO_TABLE = create_enhanced_phenotype_table(
    PHENO_GUIDE, CONTRIB_GUIDE, W_VALUES_AVAILABLE, LOGW_MAT_TL
)

DEFAULT_PHENOTYPE = "Alzheimer's disease/dementia"
plots = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            'pheno_select',
            'Choose a phenotype:',
            ALL_PHENOTYPES,
            selected=DEFAULT_PHENOTYPE,
        ),
        ui.input_slider(
            'n_var_components', 'Number of Latent Factors', min=1, max=10, value=5, step=1.0
        ),
        ui.input_slider(
            'n_phenotypes_show', 'Number of Phenotypes in Plot', min=1, max=5, value=3, step=1.0
        ),
        ui.input_slider('topn_loci', 'Number of Loci in Plot', min=1, max=15, value=9, step=1.0),
        ui.input_slider(
            'topn_genes', 'Number of Example Genes per Locus', min=1, max=5, value=2, step=1.0
        ),
        ui.download_button('download_genes', 'Download Gene List'),
        width=300,
    ),
    ui.HTML(
        """
    <h3>GUIDE Browser - Bar Plots</h3>
    <p>Select a phenotype of interest to visualize variance components.</p>
    <p><strong>Reference:</strong> <a href="https://www.biorxiv.org/content/10.1101/2024.05.03.592285v2" target="_blank">Lazarev et al. bioRxiv 2024</a></p>
    """
    ),
    ui.row(
        ui.column(
            3,
            ui.output_plot('plot_variance_components_controller'),
        ),
        ui.column(9, ui.output_plot('pheno_contrib_to_lat_controller')),
    ),
    ui.row(ui.output_plot('var_contrib_to_lat_controller', height=f'{PLOT_HEIGHT}px')),
)

table = ui.page_fluid(
    ui.HTML(
        """
    <h3>GUIDE Browser - Phenotype Table</h3>
    <p>Each phenotype row shows its top 3 associated latent factors with variance components and w-values.</p>
    <p><strong>Reference:</strong> <a href="https://www.biorxiv.org/content/10.1101/2024.05.03.592285v2" target="_blank">Lazarev et al. bioRxiv 2024</a></p>
    """
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_text('filter_pheno', 'Filter by phenotype:', placeholder='Type to search...'),
            ui.input_text('filter_latent', 'Filter by latent:', placeholder='e.g., Lat42'),
            ui.input_numeric(
                'min_contrib', 'Min variance component:', value=0, min=0, max=1, step=0.01
            ),
            ui.hr(),
            ui.download_button('download_table', 'Download Table'),
            width=280,
        ),
        ui.card(
            ui.card_header('Phenotype Association Table'),
            ui.output_ui('custom_table'),
            full_screen=True,
            height='800px',
        ),
    ),
    class_='p-3',
)

manhattan_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            'manhattan_trait', 'Select Trait:', choices=ALL_PHENOTYPES, selected=DEFAULT_PHENOTYPE
        ),
        ui.hr(),
        ui.markdown('**Chromosomal View**'),
        ui.input_checkbox('manhattan_regional', 'Enable Chromosomal View', value=False),
        ui.panel_conditional(
            'input.manhattan_regional',
            ui.input_numeric('manhattan_chr', 'Chromosome:', value=1, min=1, max=22),
            ui.input_numeric('manhattan_start', 'Start Position (bp):', value=1000000, min=1),
            ui.input_numeric('manhattan_end', 'End Position (bp):', value=2000000, min=1),
        ),
        ui.hr(),
        ui.download_button('download_manhattan_data', 'Download Data'),
        width=300,
    ),
    ui.markdown(
        """
    <h3>Manhattan Plots</h3>
    <p>Select a trait to view its variant associations.
    Variants are colored by which of the trait's top 3 latent factors they load onto:</p>
    <ul>
        <li><strong>Red</strong>: Loads onto 1st latent (En = enrichment score)</li>
        <li><strong>Orange</strong>: Loads onto 2nd latent</li>
        <li><strong>Green</strong>: Loads onto 3rd latent</li>
        <li><strong>Gray</strong>: Loads onto other latents</li>
    </ul>
    <p>Enrichment score: En = (g_sig/g_total) / (n_sig/n_total)</p>
    <p><strong>Reference:</strong> <a href="https://www.biorxiv.org/content/10.1101/2024.05.03.592285v2" target="_blank">Lazarev et al. bioRxiv 2024</a></p>
    """
    ),
    ui.output_plot('manhattan_plot', height='600px'),
    ui.output_ui('manhattan_info_ui'),
)

llm_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            'llm_latent_select',
            'Select Latent Factor:',
            choices=LLM_LATENT_CHOICES,
            selected=list(LLM_LATENT_CHOICES.keys())[0] if LLM_LATENT_CHOICES else None,
        ),
        ui.hr(),
        ui.download_button('download_llm_data', 'Download Factor Data'),
        width=320,
    ),
    ui.HTML(
        """
    <h3>GUIDE Browser - LLM Characterization</h3>
    <p><strong>Reference:</strong> <a href="https://www.biorxiv.org/content/10.1101/2024.05.03.592285v2" target="_blank">Lazarev et al. bioRxiv 2024</a></p>
    """
    ),
    ui.output_ui('llm_factor_display'),
    ui.HTML(f"""
    <details style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
        <summary style="cursor: pointer; font-weight: 600; font-size: 14px; color: #495057; user-select: none;">
            <span style="margin-left: 8px;">About this Dataset and LLM Characterization</span>
        </summary>
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #dee2e6; line-height: 1.6; font-size: 13px; color: #495057;">
            <h4 style="margin-top: 0; color: #495057; font-size: 14px;">Dataset</h4>
            <p style="margin-bottom: 12px;">
                This browser uses the "all" dataset from
                <a href="https://www.nature.com/articles/s41467-019-11953-9" target="_blank" style="color: #0d6efd;">Tanigawa et al. (Nature Communications 2019)</a>,
                which includes comprehensive phenotypic data from the UK Biobank.
            </p>

            <h4 style="color: #495057; font-size: 14px; margin-top: 16px;">Creating Your Own Model</h4>
            <p style="margin-bottom: 12px;">
                To create your own GUIDE model, visit the
                <a href="https://github.com/daniel-lazarev/GUIDE" target="_blank" style="color: #0d6efd;">GUIDE GitHub repository</a>
                for the relevant code. For detailed methodology and theoretical background, refer to our paper:
                <a href="https://www.biorxiv.org/content/10.1101/2024.05.03.592285v2" target="_blank" style="color: #0d6efd;">Lazarev et al. bioRxiv 2024</a>.
            </p>

            <h4 style="color: #495057; font-size: 14px; margin-top: 16px;">LLM Characterization Details</h4>
            <p style="margin-bottom: 12px;">
                The latent factor characterizations displayed in this browser were generated using
                <strong>Claude Opus 4.5</strong> with extended thinking enabled and a minimized temperature parameter to ensure reproducibility.
                The model analyzed the top associated phenotypes, genetic variants, and biological pathways for each latent factor
                to produce interpretable summaries of the underlying biological mechanisms.
            </p>

            <details style="margin-top: 16px; padding: 12px; background-color: white; border-radius: 4px; border: 1px solid #dee2e6;">
                <summary style="cursor: pointer; font-weight: 600; font-size: 13px; color: #667eea; user-select: none;">
                    <span style="margin-left: 8px;">View Full LLM Prompt</span>
                </summary>
                <div style="margin-top: 12px; max-height: 400px; overflow-y: auto;">
                    <pre style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; font-size: 11px; line-height: 1.4; margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: 'Courier New', monospace;">{LLM_PROMPT}</pre>
                </div>
            </details>
        </div>
    </details>
    """),
)

app_ui = ui.page_navbar(
    ui.nav_panel('Table', table),
    ui.nav_panel('Bar Plots', plots),
    ui.nav_panel('Manhattan Plots', manhattan_ui),
    ui.nav_panel('LLM Characterization', llm_ui),
    title='GUI for GUIDE',
    id='main_navbar',
    header=ui.tags.script("""
        function navigateToTab(tabName) {
            const tabs = document.querySelectorAll('.nav-link');
            tabs.forEach(tab => {
                if (tab.textContent.trim() === tabName) {
                    tab.click();
                }
            });
        }

        function navigateToManhattan(phenotype) {
            navigateToTab('Manhattan Plots');
            setTimeout(() => {
                Shiny.setInputValue('nav_phenotype', phenotype, {priority: 'event'});
            }, 100);
        }

        function navigateToBarPlots(phenotype) {
            navigateToTab('Bar Plots');
            setTimeout(() => {
                Shiny.setInputValue('nav_phenotype', phenotype, {priority: 'event'});
            }, 100);
        }

        function navigateToLLM(latentKey) {
            navigateToTab('LLM Characterization');
            setTimeout(() => {
                Shiny.setInputValue('nav_latent', latentKey, {priority: 'event'});
            }, 100);
        }
    """),
)


def server(input: Inputs, output: Outputs, session: Session):

    selected_phenotype = reactive.value(DEFAULT_PHENOTYPE)
    nav_latent_value = reactive.value(None)

    @reactive.effect
    @reactive.event(input.nav_phenotype)
    def handle_nav_phenotype():
        pheno = input.nav_phenotype()
        if pheno and pheno in ALL_PHENOTYPES:
            selected_phenotype.set(pheno)

    @reactive.effect
    @reactive.event(input.nav_latent)
    def handle_nav_latent():
        latent = input.nav_latent()
        print(f'>>> NAV_LATENT RECEIVED: {latent}')
        if latent:
            latent_key = f'Lat{latent}'
            print(f'>>> LOOKING FOR KEY: {latent_key}')
            if latent_key in LLM_LATENT_CHOICES:
                print(f'>>> UPDATING DROPDOWN TO: {latent_key}')
                ui.update_selectize('llm_latent_select', selected=latent_key)
            else:
                print('>>> KEY NOT FOUND IN CHOICES')

    @reactive.effect
    @reactive.event(input.pheno_select)
    def sync_from_barplots():
        pheno = input.pheno_select()
        print(f'=== BARPLOTS CHANGED: {pheno}')
        if pheno:
            selected_phenotype.set(pheno)
            print(f'=== SET selected_phenotype to: {pheno}')

    @reactive.effect
    @reactive.event(input.manhattan_trait)
    def sync_from_manhattan():
        pheno = input.manhattan_trait()
        if pheno and pheno != selected_phenotype.get():
            selected_phenotype.set(pheno)

    @reactive.effect
    def update_pheno_select():
        pheno = selected_phenotype.get()
        if pheno:
            ui.update_selectize('pheno_select', selected=pheno)

    @reactive.effect
    def update_manhattan_trait():
        pheno = selected_phenotype.get()
        if pheno:
            ui.update_selectize('manhattan_trait', selected=pheno)

    @reactive.calc
    def current_llm_latent():
        dropdown_val = input.llm_latent_select()
        nav_val = nav_latent_value.get()
        if nav_val and dropdown_val != nav_val:
            return nav_val
        return dropdown_val

    @reactive.effect
    def clear_nav_latent_after_sync():
        dropdown_val = input.llm_latent_select()
        nav_val = nav_latent_value.get()
        if nav_val and dropdown_val == nav_val:
            nav_latent_value.set(None)

    @output
    @render.plot
    def plot_variance_components_controller():
        interested_phenos = selected_phenotype.get()
        if isinstance(interested_phenos, str):
            interested_phenos = [interested_phenos]
        paths = [
            f'./all_phenos/phenotypes/guide/{get_safe_phe_name(p)}/squared_cosine_scores.tsv'
            for p in interested_phenos
        ]
        df = pd.read_csv(paths[0], sep='\t')

        topn_var = input.n_var_components()
        fig, ax, latent_labels_local = plot_variance_components(
            df, ['squared_cosine_score'], [interested_phenos], topn=topn_var, bar_width=0.0001
        )
        set_latent_labels(latent_labels_local)
        return fig

    @output
    @render.plot
    def var_contrib_to_lat_controller():
        if not LATENT_LABELS:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(
                0.5,
                0.5,
                'Please select a phenotype first',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
            ax.axis('off')
            return fig
        interested_phenos = selected_phenotype.get()
        if isinstance(interested_phenos, str):
            interested_phenos = [interested_phenos]
        paths = [
            f'./all_phenos/phenotypes/guide/{get_safe_phe_name(p)}/squared_cosine_scores.tsv'
            for p in interested_phenos
        ]
        fig, ax = var_contrib_to_lat(
            CYTO_CONTRIB_GUIDE,
            LATENT_LABELS,
            spacing=3.5,
            path_to_rel_df=paths,
            topn=input.topn_loci(),
            topn_genes=input.topn_genes(),
            read_colors=True,
            gene2cyto_dict=GENE2CYTO_DICT_GUIDE,
        )
        return fig

    @output
    @render.plot
    def pheno_contrib_to_lat_controller():
        # interested_phenos = selected_phenotype.get()  # FIXME: unused. bug?
        if not LATENT_LABELS:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(
                0.5,
                0.5,
                'Please select a phenotype first',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
            ax.axis('off')
            return fig
        df = gen_barplot_data(PHENO_GUIDE, CONTRIB_GUIDE, True, COLORS)
        input.topn_genes()
        input.topn_loci()
        fig = pheno_contribs_to_lat(df, LATENT_LABELS, topn=input.n_phenotypes_show(), spacing=3)
        return fig

    @session.download(
        filename=lambda: (
            f'guide-genes-{LATENT_LABELS[0].replace(" ", "_") if LATENT_LABELS else "none"}-{date.today().isoformat()}.csv'
        )
    )
    async def download_genes():
        if not LATENT_LABELS:
            yield 'Error: No latent factors selected\n'
            return
        yield 'Gene_Name\tVariance_Component\n'
        for key, value in CYTO_CONTRIB_GUIDE[LATENT_LABELS[0]].sort_values(ascending=False).items():
            yield f'{key}\t{value}\n'

    @reactive.Calc
    def filtered_table():
        df = ENHANCED_PHENO_TABLE.copy()
        pheno_filter = input.filter_pheno().strip()
        if pheno_filter:
            df = df[df['Phenotype'].str.contains(pheno_filter, case=False, na=False)]
        latent_filter = input.filter_latent().strip()
        if latent_filter:
            mask = df['latent_info'].apply(
                lambda info_list: any(
                    latent_filter.lower() in item['latent'].lower() for item in info_list
                )
            )
            df = df[mask]
        min_contrib = input.min_contrib()
        if min_contrib > 0:
            mask = df['latent_info'].apply(
                lambda info_list: any(item['contribution'] >= min_contrib for item in info_list)
            )
            df = df[mask]
        return df

    @output
    @render.ui
    def custom_table():
        df = filtered_table()
        if len(df) == 0:
            return ui.HTML(
                "<p style='text-align: center; padding: 40px; color: #999;'>No phenotypes match your filters.</p>"
            )
        html_parts = [
            """
        <style>
            .pheno-table {
                width: 100%;
                border-collapse: collapse;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 13px;
            }
            .pheno-table thead {
                background-color: #f8f9fa;
                border-bottom: 2px solid #dee2e6;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .pheno-table th {
                padding: 12px 16px;
                text-align: left;
                font-weight: 600;
                color: #495057;
                border-bottom: 2px solid #dee2e6;
            }
            .pheno-table td {
                padding: 8px 16px;
                border-bottom: 1px solid #e9ecef;
                vertical-align: top;
            }
            .pheno-table tbody tr:hover {
                background-color: #f8f9fa;
            }
            .pheno-name {
                font-weight: 500;
                color: #212529;
                max-width: 300px;
            }
            .pheno-name-link {
                color: #0d6efd;
                cursor: pointer;
                text-decoration: none;
            }
            .pheno-name-link:hover {
                text-decoration: underline;
            }
            .latent-subtable {
                width: 100%;
                border-collapse: collapse;
            }
            .latent-subtable tr {
                border-bottom: 1px solid #f1f3f5;
            }
            .latent-subtable tr:last-child {
                border-bottom: none;
            }
            .latent-subtable thead tr {
                border-bottom: 1px solid #dee2e6;
                background-color: #f8f9fa;
            }
            .latent-subtable thead td {
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
                color: #6c757d;
                border: none;
            }
            .latent-subtable tbody td {
                padding: 4px 8px;
                border: none;
            }
            .latent-subtable td {
                padding: 4px 8px;
                border: none;
            }
            .latent-col {
                width: 100px;
                font-family: "Courier New", monospace;
                font-weight: 500;
            }
            .latent-link {
                color: #0d6efd;
                cursor: pointer;
                text-decoration: none;
            }
            .latent-link:hover {
                text-decoration: underline;
            }
            .contrib-col {
                width: 120px;
                text-align: right;
                font-family: "Courier New", monospace;
            }
            .w-col {
                width: 100px;
                text-align: right;
                font-family: "Courier New", monospace;
            }
        </style>
        <div style="overflow-y: auto; max-height: 700px;">
        <table class="pheno-table">
            <thead>
                <tr>
                    <th style="width: 300px;">Phenotype</th>
                    <th>Top 3 Latent Factors</th>
                </tr>
            </thead>
            <tbody>
        """
        ]
        for _, row in df.iterrows():
            pheno_name = row['Phenotype']
            pheno_escaped = pheno_name.replace("'", "\\'").replace('"', '&quot;')
            latent_info = row['latent_info']
            html_parts.append(f"""
                <tr>
                    <td class="pheno-name">
                        <span class="pheno-name-link" onclick="navigateToBarPlots('{pheno_escaped}')">{pheno_name}</span>
                    </td>
                    <td>
                        <table class="latent-subtable">
                            <thead>
                                <tr style="font-size: 11px; color: #6c757d; font-weight: 600;">
                                    <td class="latent-col">Latent</td>
                                    <td class="contrib-col">Variance Component</td>
                                    <td class="w-col">-log10(w)</td>
                                </tr>
                            </thead>
                            <tbody>
            """)
            for info in latent_info:
                latent = info['latent']
                latent_idx = info['latent_idx']
                latent_key = str(latent_idx)
                contrib = info['contribution']
                w_val = info['w_value']
                w_str = f'{w_val:.4f}' if not pd.isna(w_val) else 'N/A'
                html_parts.append(f"""
                        <tr>
                            <td class="latent-col"><span class="latent-link" onclick="navigateToLLM('{latent_key}')">{latent}</span></td>
                            <td class="contrib-col">{contrib:.4f}</td>
                            <td class="w-col">{w_str}</td>
                        </tr>
                """)
            html_parts.append("""
                            </tbody>
                        </table>
                    </td>
                </tr>
            """)
        html_parts.append("""
            </tbody>
        </table>
        </div>
        <div style="margin-top: 16px; padding: 12px; background-color: #f8f9fa; border-radius: 4px; font-size: 12px; color: #6c757d;">
            <strong>Note:</strong>
            Latents are ordered by variance component (highest to lowest for each phenotype).
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Columns: Latent ID | Variance Component | -log10(w)
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <em>Phenotypes sorted by cumulative top 3 variance component (highest first)</em>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <em>Click phenotype names to view Bar Plots, click latents to view LLM characterization</em>
        </div>
        """)
        return ui.HTML(''.join(html_parts))

    @session.download(filename=lambda: f'guide-phenotype-table-{date.today().isoformat()}.csv')
    async def download_table():
        df = filtered_table()
        yield 'Phenotype,Latent,Variance_Component,w_value\n'
        for _, row in df.iterrows():
            pheno = row['Phenotype']
            for info in row['latent_info']:
                latent = info['latent']
                contrib = info['contribution']
                w_val = info['w_value']
                w_str = f'{w_val:.6f}' if not pd.isna(w_val) else 'NA'
                yield f'"{pheno}",{latent},{contrib:.6f},{w_str}\n'

    @output
    @render.plot
    def manhattan_plot():
        trait = selected_phenotype.get()
        if not trait:
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.text(
                0.5,
                0.5,
                'Please select a trait',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.axis('off')
            return fig
        chr_filter = None
        region_start = None
        region_end = None
        if input.manhattan_regional():
            chr_filter = input.manhattan_chr()
            region_start = input.manhattan_start()
            region_end = input.manhattan_end()
            if region_start >= region_end:
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.text(
                    0.5,
                    0.5,
                    'Error: Start position must be less than end position',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=14,
                    color='red',
                )
                ax.axis('off')
                return fig
        selected_trait = trait
        df = prepare_manhattan_data(
            VAR_LABELS,
            CONTRIB_VAR_GUIDE,
            selected_trait,
            CONTRIB_GUIDE,
            var_gene_file='./snp_gene_pq.txt',
        )
        if df is None:
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.text(
                0.5,
                0.5,
                'Error: Could not load data for trait',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=14,
                color='red',
            )
            ax.axis('off')
            return fig
        top3_latent_labels = df.attrs.get('top3_latent_labels', [])
        if chr_filter:
            title = f'{selected_trait}\nChr{chr_filter}:{region_start:,}-{region_end:,}'
        else:
            title = f'{selected_trait}'
        if top3_latent_labels:
            title += f'\nTop latents: {", ".join(top3_latent_labels)}'
        fig, ax = plot_manhattan(
            df,
            value_col='value',
            title=title,
            chr_filter=chr_filter,
            region_start=region_start,
            region_end=region_end,
            use_pvalues=ZSCORES_AVAILABLE,
            figsize=(16, 6) if not chr_filter else (14, 6),
        )
        return fig

    @output
    @render.ui
    def manhattan_info_ui():
        trait = selected_phenotype.get()
        if not trait:
            return ui.markdown('**Please select a trait.**')
        selected_trait = trait
        df = prepare_manhattan_data(VAR_LABELS, CONTRIB_VAR_GUIDE, selected_trait, CONTRIB_GUIDE)
        if df is None:
            return ui.markdown('**Error: Could not load data for trait.**')
        top3_latent_labels = df.attrs.get('top3_latent_labels', [])
        top3_latents = df.attrs.get('top3_latents', [])
        top3_contributions = df.attrs.get('top3_contributions', [])
        top3_w_values = df.attrs.get('top3_w_values', [])
        if input.manhattan_regional():
            chr_filter = input.manhattan_chr()
            region_start = input.manhattan_start()
            region_end = input.manhattan_end()
            df = df[df['CHR'] == chr_filter]
            df = df[(df['POS'] >= region_start) & (df['POS'] <= region_end)]
        n_variants = len(df)
        if n_variants == 0:
            return ui.markdown('**No variants in selected region.**')
        if ZSCORES_AVAILABLE:
            threshold = 7.3
        else:
            threshold = df['value'].quantile(0.95)
        n_above_threshold = (df['value'] > threshold).sum()
        max_val = df['value'].max()
        mean_val = df['value'].mean()
        n_1st = (df['top_latent_rank'] == 1).sum()
        n_2nd = (df['top_latent_rank'] == 2).sum()
        n_3rd = (df['top_latent_rank'] == 3).sum()
        n_other = (df['top_latent_rank'] == 0).sum()
        ntotal = len(df)
        nsig = n_above_threshold
        enrichment_scores = {}
        for rank, count in [(1, n_1st), (2, n_2nd), (3, n_3rd), (0, n_other)]:
            mask = df['top_latent_rank'] == rank
            gtotal = mask.sum()
            gsig = ((df['value'] > threshold) & mask).sum()
            if gtotal > 0 and ntotal > 0 and nsig > 0:
                enrichment = (gsig / gtotal) / (nsig / ntotal)
            else:
                enrichment = 1.0
            enrichment_scores[rank] = enrichment
        top_variants = df.nlargest(5, 'value')
        value_label = '-log10(p-value)' if ZSCORES_AVAILABLE else 'Weighted Variance Component'
        info_parts = [
            f'### {selected_trait}',
            '',
            '### Top 3 Contributing Latent Factors',
        ]

        for i, (lat_label, lat_idx, contrib, w_val) in enumerate(
            zip(top3_latent_labels, top3_latents, top3_contributions, top3_w_values), 1
        ):
            latent_key = str(lat_idx)
            if not pd.isna(w_val):
                info_parts.append(
                    f'{i}. <span class="latent-link" onclick="navigateToLLM(\'{latent_key}\')" style="color: #0d6efd; cursor: pointer;">**{lat_label}**</span> (variance component: {contrib:.4f}, -log10(w-value): {w_val:.4f})'
                )
            else:
                info_parts.append(
                    f'{i}. <span class="latent-link" onclick="navigateToLLM(\'{latent_key}\')" style="color: #0d6efd; cursor: pointer;">**{lat_label}**</span> (variance component: {contrib:.4f})'
                )

        info_parts.extend(
            [
                '',
                '### Summary Statistics',
                f'- **Total variants displayed:** {n_variants:,}',
                f'- **Variants above threshold ({threshold:.2f}):** {n_above_threshold:,}',
                f'- **Maximum {value_label}:** {max_val:.4f}',
                f'- **Mean {value_label}:** {mean_val:.6f}',
                '',
                '### Variants by Latent Factor Loading',
                f'- **{top3_latent_labels[0] if len(top3_latent_labels) > 0 else "1st"} (Red):** {n_1st:,} variants (En={enrichment_scores[1]:.2f})',
                f'- **{top3_latent_labels[1] if len(top3_latent_labels) > 1 else "2nd"} (Orange):** {n_2nd:,} variants (En={enrichment_scores[2]:.2f})',
                f'- **{top3_latent_labels[2] if len(top3_latent_labels) > 2 else "3rd"} (Green):** {n_3rd:,} variants (En={enrichment_scores[3]:.2f})',
                f'- **Other (Gray):** {n_other:,} variants (En={enrichment_scores[0]:.2f})',
                '',
                'Note: values below are capped at -log10(w-value) or -log10(p-value) = 300',
                '',
                '### Top 5 Variants by Association',
                '',
            ]
        )

        for i, (_, row) in enumerate(top_variants.iterrows(), 1):
            gene_info = f' ({row["GENE"]})' if row['GENE'] else ''
            rank = int(row['top_latent_rank'])
            if rank > 0 and len(top3_latent_labels) >= rank:
                rank_info = f' [{top3_latent_labels[rank - 1]}]'
            else:
                rank_info = ''

            info_parts.append(
                f'{i}. **{row["VAR"]}**{gene_info}{rank_info}  \n'
                f'   {value_label}: {row["value"]:.4f}'
            )
        return ui.HTML(ui.markdown('\n'.join(info_parts)))

    @session.download(
        filename=lambda: (
            f'manhattan-{selected_phenotype.get().replace("/", "_").replace(" ", "_") if selected_phenotype.get() else "data"}-{date.today().isoformat()}.csv'
        )
    )
    async def download_manhattan_data():
        trait = selected_phenotype.get()
        if not trait:
            yield 'Error: No trait selected\n'
            return
        selected_trait = trait
        df = prepare_manhattan_data(VAR_LABELS, CONTRIB_VAR_GUIDE, selected_trait, CONTRIB_GUIDE)
        if df is None:
            yield 'Error: Could not load data for trait.\n'
            return
        top3_latent_labels = df.attrs.get('top3_latent_labels', [])
        if input.manhattan_regional():
            chr_filter = input.manhattan_chr()
            region_start = input.manhattan_start()
            region_end = input.manhattan_end()
            df = df[df['CHR'] == chr_filter]
            df = df[(df['POS'] >= region_start) & (df['POS'] <= region_end)]
        df = df.sort_values('value', ascending=False)
        value_label = '-log10(p)' if ZSCORES_AVAILABLE else 'WeightedVarianceComponent'
        if ZSCORES_AVAILABLE:
            threshold = 7.3
        else:
            threshold = df['value'].quantile(0.95)
        ntotal = len(df)
        nsig = (df['value'] > threshold).sum()
        yield '# Manhattan Plots Data\n'
        yield f'# Trait: {selected_trait}\n'
        yield f'# Top 3 Latents: {", ".join(top3_latent_labels)}\n'
        yield f'# Generated: {date.today().isoformat()}\n'
        yield f'# Threshold: {threshold:.2f}\n'
        yield '#\n'
        if W_VALUES_AVAILABLE:
            yield f'Chromosome\tPosition\tVariant\tGene\t{value_label}\tTopLatent\tEnrichment\tw_value\n'
        else:
            yield f'Chromosome\tPosition\tVariant\tGene\t{value_label}\tTopLatent\tEnrichment\n'
        for _, row in df.iterrows():
            rank = int(row['top_latent_rank'])
            if rank > 0 and len(top3_latent_labels) >= rank:
                latent_str = top3_latent_labels[rank - 1]
            else:
                latent_str = 'Other'
            mask = df['top_latent_rank'] == rank
            gtotal = mask.sum()
            gsig = ((df['value'] > threshold) & mask).sum()
            if gtotal > 0 and ntotal > 0 and nsig > 0:
                enrichment = (gsig / gtotal) / (nsig / ntotal)
            else:
                enrichment = 1.0
            if W_VALUES_AVAILABLE:
                w_val_str = f'{row["w_value"]:.8f}' if not pd.isna(row['w_value']) else 'NA'
                yield f'{int(row["CHR"])}\t{int(row["POS"])}\t{row["VAR"]}\t{row["GENE"]}\t{row["value"]:.8f}\t{latent_str}\t{enrichment:.4f}\t{w_val_str}\n'
            else:
                yield f'{int(row["CHR"])}\t{int(row["POS"])}\t{row["VAR"]}\t{row["GENE"]}\t{row["value"]:.8f}\t{latent_str}\t{enrichment:.4f}\n'

    @output
    @render.ui
    def llm_factor_display():
        current_latent = input.llm_latent_select()  # This is now "Lat39"
        print(f'>>> RENDERING WITH LATENT: {current_latent}')

        if not current_latent or current_latent not in LLM_DATA.get('latent_factors', {}):
            return ui.HTML('<p>Please select a latent factor.</p>')

        factor = LLM_DATA['latent_factors'][current_latent]
        factor_num = current_latent.replace('Lat', '')

        html_parts = [
            """
        <style>
            .llm-container {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                max-width: 1200px;
                padding: 20px;
            }
            .llm-header {
                background-color: #667eea;
                color: white;
                padding: 24px;
                border-radius: 12px;
                margin-bottom: 24px;
            }
            .llm-header h2 {
                margin: 0 0 8px 0;
                font-size: 28px;
            }
            .llm-header .mechanism {
                font-size: 16px;
                opacity: 0.9;
            }
            .llm-section {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .llm-section h3 {
                color: #495057;
                border-bottom: 2px solid #667eea;
                padding-bottom: 8px;
                margin-top: 0;
                margin-bottom: 16px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: 700;
                color: #667eea;
            }
            .stat-label {
                font-size: 12px;
                color: #6c757d;
                text-transform: uppercase;
                margin-top: 4px;
            }
            .bio-interpretation {
                line-height: 1.7;
                color: #343a40;
                font-size: 15px;
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }
            .data-table th {
                background: #f8f9fa;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                color: #495057;
                border-bottom: 2px solid #dee2e6;
                position: sticky;
                top: 0;
            }
            .data-table td {
                padding: 10px 12px;
                border-bottom: 1px solid #e9ecef;
            }
            .data-table tbody tr:hover {
                background-color: #f8f9fa;
            }
            .data-table .rank-col {
                width: 50px;
                text-align: center;
                font-weight: 600;
                color: #667eea;
            }
            .data-table .value-col {
                text-align: right;
                font-family: "Courier New", monospace;
            }
            .pheno-link {
                color: #0d6efd;
                cursor: pointer;
                text-decoration: none;
            }
            .pheno-link:hover {
                text-decoration: underline;
            }
            .gene-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .gene-tag {
                background: #e7f1ff;
                color: #0d6efd;
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 13px;
                font-weight: 500;
            }
            .region-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .region-tag {
                background: #fff3cd;
                color: #856404;
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 13px;
                font-weight: 500;
            }
            .annotation-item {
                padding: 12px;
                border-left: 3px solid #667eea;
                background: #f8f9fa;
                margin-bottom: 8px;
                border-radius: 0 8px 8px 0;
            }
            .annotation-gene {
                font-weight: 600;
                color: #495057;
            }
            .annotation-function {
                color: #6c757d;
                font-size: 14px;
                margin-top: 4px;
            }
            .table-container {
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #e9ecef;
                border-radius: 8px;
            }
        </style>
        """
        ]

        label = factor.get('label', 'Unknown')
        mechanism = factor.get('primary_mechanism', 'Not specified')

        html_parts.append(f"""
        <div class="llm-container">
            <div class="llm-header">
                <h2>Latent Factor {factor_num}: {label}</h2>
                <div class="mechanism"><strong>Primary Mechanism:</strong> {mechanism}</div>
            </div>
        """)

        stats = factor.get('summary_statistics', {})
        html_parts.append("""
            <div class="llm-section">
                <h3>Summary Statistics</h3>
                <div class="stats-grid">
        """)

        if stats.get('top_phenotype_variance_component') is not None:
            html_parts.append(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['top_phenotype_variance_component']:.4f}</div>
                        <div class="stat-label">Top Phenotype Variance Component</div>
                    </div>
            """)

        if stats.get('top_phenotype_neg_log10_w_value') is not None:
            html_parts.append(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['top_phenotype_neg_log10_w_value']:.2f}</div>
                        <div class="stat-label">Top Phenotype -log10(w-value)</div>
                    </div>
            """)

        primary_domain = stats.get('primary_domain')
        if primary_domain:
            html_parts.append(f"""
                    <div class="stat-card">
                        <div class="stat-value">{primary_domain['variance_component_percentage']:.1f}%</div>
                        <div class="stat-label">Primary Domain: {primary_domain['name']}</div>
                    </div>
            """)

        secondary_domain = stats.get('secondary_domain')
        if secondary_domain:
            html_parts.append(f"""
                    <div class="stat-card">
                        <div class="stat-value">{secondary_domain['variance_component_percentage']:.1f}%</div>
                        <div class="stat-label">Secondary Domain: {secondary_domain['name']}</div>
                    </div>
            """)

        html_parts.append("""
                </div>
            </div>
        """)

        bio_interp = factor.get('biological_interpretation', '')
        if bio_interp:
            html_parts.append(f"""
            <div class="llm-section">
                <h3>Biological Interpretation</h3>
                <p class="bio-interpretation">{bio_interp}</p>
            </div>
            """)

        phenos_var = factor.get('top_phenotypes_by_variance', [])
        if phenos_var:
            html_parts.append("""
            <div class="llm-section">
                <h3>Top Phenotypes by Variance Component</h3>
                <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th class="rank-col">Rank</th>
                            <th>Phenotype</th>
                            <th class="value-col">Variance Component</th>
                        </tr>
                    </thead>
                    <tbody>
            """)
            for p in phenos_var:
                pheno_name = p.get('phenotype', '')
                pheno_escaped = pheno_name.replace("'", "\\'").replace('"', '&quot;')
                if pheno_name in ALL_PHENOTYPES:
                    html_parts.append(f"""
                        <tr>
                            <td class="rank-col">{p.get('rank', '')}</td>
                            <td><span class="pheno-link" onclick="navigateToBarPlots('{pheno_escaped}')">{pheno_name}</span></td>
                            <td class="value-col">{p.get('variance_component', 0):.5f}</td>
                        </tr>
                    """)
                else:
                    html_parts.append(f"""
                        <tr>
                            <td class="rank-col">{p.get('rank', '')}</td>
                            <td>{pheno_name}</td>
                            <td class="value-col">{p.get('variance_component', 0):.5f}</td>
                        </tr>
                    """)
            html_parts.append("""
                    </tbody>
                </table>
                </div>
            </div>
            """)

        phenos_sig = factor.get('top_phenotypes_by_significance', [])
        if phenos_sig:
            html_parts.append("""
            <div class="llm-section">
                <h3>Top Phenotypes by Statistical Significance</h3>
                <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th class="rank-col">Rank</th>
                            <th>Phenotype</th>
                            <th class="value-col">-log10(w-value)</th>
                        </tr>
                    </thead>
                    <tbody>
            """)
            for p in phenos_sig:
                pheno_name = p.get('phenotype', '')
                pheno_escaped = pheno_name.replace("'", "\\'").replace('"', '&quot;')
                if pheno_name in ALL_PHENOTYPES:
                    html_parts.append(f"""
                        <tr>
                            <td class="rank-col">{p.get('rank', '')}</td>
                            <td><span class="pheno-link" onclick="navigateToBarPlots('{pheno_escaped}')">{pheno_name}</span></td>
                            <td class="value-col">{p.get('neg_log10_w_value', 0):.3f}</td>
                        </tr>
                    """)
                else:
                    html_parts.append(f"""
                        <tr>
                            <td class="rank-col">{p.get('rank', '')}</td>
                            <td>{pheno_name}</td>
                            <td class="value-col">{p.get('neg_log10_w_value', 0):.3f}</td>
                        </tr>
                    """)
            html_parts.append("""
                    </tbody>
                </table>
                </div>
            </div>
            """)

        genetics = factor.get('genetic_architecture', {})
        if genetics:
            html_parts.append("""
            <div class="llm-section">
                <h3>Genetic Architecture</h3>
            """)

            variants = genetics.get('top_genetic_variants', [])
            if variants:
                html_parts.append("""
                <h4 style="color: #6c757d; margin-top: 0;">Top Genetic Variants</h4>
                <div class="table-container" style="margin-bottom: 20px;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th class="rank-col">Rank</th>
                            <th>Variant ID</th>
                            <th>Gene</th>
                            <th>Location</th>
                            <th class="value-col">Variance Component</th>
                        </tr>
                    </thead>
                    <tbody>
                """)
                for v in variants:
                    html_parts.append(f"""
                        <tr>
                            <td class="rank-col">{v.get('rank', '')}</td>
                            <td style="font-family: monospace;">{v.get('variant_id', '')}</td>
                            <td>{v.get('gene', '')}</td>
                            <td>{v.get('location', '')}</td>
                            <td class="value-col">{v.get('variance_component', 0):.5f}</td>
                        </tr>
                    """)
                html_parts.append("""
                    </tbody>
                </table>
                </div>
                """)

            top_genes = genetics.get('top_genes', [])
            if top_genes:
                html_parts.append("""
                <h4 style="color: #6c757d;">Top Genes (by cumulative variance component)</h4>
                <div class="gene-list" style="margin-bottom: 20px;">
                """)
                for gene in top_genes:
                    html_parts.append(f'<span class="gene-tag">{gene}</span>')
                html_parts.append('</div>')

            regions = genetics.get('key_chromosomal_regions', [])
            if regions:
                html_parts.append("""
                <h4 style="color: #6c757d;">Key Chromosomal Regions</h4>
                <div class="region-list" style="margin-bottom: 20px;">
                """)
                for region in regions:
                    html_parts.append(f'<span class="region-tag">{region}</span>')
                html_parts.append('</div>')

            annotations = genetics.get('gene_function_annotations', {})
            if annotations:
                html_parts.append("""
                <h4 style="color: #6c757d;">Gene Function Annotations</h4>
                """)
                for gene, function in annotations.items():
                    html_parts.append(f"""
                <div class="annotation-item">
                    <div class="annotation-gene">{gene}</div>
                    <div class="annotation-function">{function}</div>
                </div>
                    """)

            html_parts.append('</div>')

        html_parts.append('</div>')

        return ui.HTML(''.join(html_parts))

    @session.download(
        filename=lambda: (
            f'llm-factor-{current_llm_latent().replace("Lat", "") if current_llm_latent() else "data"}-{date.today().isoformat()}.json'
        )
    )
    async def download_llm_data():
        nav_val = nav_latent_value.get()
        dropdown_val = input.llm_latent_select()
        current_latent = nav_val if nav_val else dropdown_val
        if not current_latent:
            yield '{"error": "No data available"}'
            return
        if current_latent in LLM_DATA.get('latent_factors', {}):
            factor_data = {
                'latent_factor': current_latent,
                'data': LLM_DATA['latent_factors'][current_latent],
            }
            yield json.dumps(factor_data, indent=2)
        else:
            yield '{"error": "Factor not found"}'

    @reactive.effect
    def debug_selected_phenotype():
        val = selected_phenotype.get()
        print(f'*** selected_phenotype IS NOW: {val}')

    @reactive.effect
    @reactive.event(input.llm_latent_select)
    def debug_llm_select():
        val = input.llm_latent_select()
        print(f'>>> LLM DROPDOWN CHANGED TO: {val}')
        print(nav_latent_value.get())


app = App(app_ui, server, debug=True)
