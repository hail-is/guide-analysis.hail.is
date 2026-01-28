import csv
import itertools as it

import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from datetime import date
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import norm

from plotting import var_contrib_to_lat, plot_variance_components, pheno_contribs_to_lat

ALL_LATENT_LABELS = [f'GUIDE Lat{i}' for i in range(100)]
LATENT_LABELS = []
PLOT_HEIGHT = 400


def load_w_values():
    try:
        w_values_data = np.load('./w_values.npz')
        logw_mat_TL = w_values_data['logw_mat_TL']
        logw_mat_XL = w_values_data['logw_mat_XL']
        print('w-values loaded successfully!')
        return True, logw_mat_TL, logw_mat_XL
    except (FileNotFoundError, KeyError):
        return False, None, None


W_VALUES_AVAILABLE, LOGW_MAT_TL, LOGW_MAT_XL = load_w_values()


def load_log10pvalues():
    # there are around 3.8GiB of zscores/pvalues, it's read only so we can
    # memory map it in order to run in a more constrained environment.
    try:
        log10pvalues = np.load('./log10pvalues.npy', mmap_mode='r')
    except FileNotFoundError:
        log10pvalues = None

    if log10pvalues is None:
        # getting here means we haven't precomputed the log10 pvalues,
        # so we try to do so, and eat the memory usage
        try:
            zscores = np.load('./degas_betas.npy')

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
        except Exception as e:
            print(f'Could not load z-scores: {e}')
            return False, None

    print(f'Z-scores loaded successfully! Shape: {log10pvalues.shape}')
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
    print(
        f'DEBUG: value column min={data_df[value_col].min():.4f}, max={data_df[value_col].max():.4f}'
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
            label = f'{top3_latent_labels[rank - 1]} (E={enrichment_scores[rank]:.2f})'
        else:
            label = f'Other (E={enrichment_scores[rank]:.2f})'
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
    print(f'DEBUG: Setting ylim to [{y_min}, {y_max + y_padding}]')
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
                    'contribution': contrib_val,
                    'w_value': w_val,
                }
            )
        rows.append({'Phenotype': pheno, 'latent_info': latent_info, 'top3_total': top3_total})
    df = pd.DataFrame(rows)
    df = df.sort_values('top3_total', ascending=False).reset_index(drop=True)
    return df


GZ = np.load('./guide_all_100lat_bl_ll.npz', allow_pickle=True)
COS_GUIDE = GZ['cos_phe']
PHENO_GUIDE = GZ['label_phe']
FACTOR_GUIDE = GZ['factor_phe']
CONTRIB_GUIDE = GZ['contribution_phe']
GENE_CONTRIB_GUIDE = GZ['contribution_gene']
CONTRIB_VAR_GUIDE = GZ['contribution_var']
VAR_LABELS = GZ['label_var']
del GZ

COLORS = list(pd.read_csv('./colors.txt', sep='\t', header=None)[0])

CYTO_CONTRIB_GUIDE, GENE2CYTO_DICT_GUIDE = gen_data(VAR_LABELS, CONTRIB_VAR_GUIDE)
ENHANCED_PHENO_TABLE = create_enhanced_phenotype_table(
    PHENO_GUIDE, CONTRIB_GUIDE, W_VALUES_AVAILABLE, LOGW_MAT_TL
)

plots = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            'pheno_select',
            'Choose a phenotype:',
            list(PHENO_GUIDE),
            selected="Alzheimer's disease/dementia",
        ),
        ui.input_slider(
            'n_var_components',
            'Number of Latent Factors',
            min=1,
            max=10,
            value=5,
            step=1.0,
        ),
        ui.input_slider(
            'n_phenotypes_show',
            'Number of Phenotypes in Plot',
            min=1,
            max=5,
            value=3,
            step=1.0,
        ),
        ui.input_slider('topn_loci', 'Number of Loci in Plot', min=1, max=15, value=9, step=1.0),
        ui.input_slider(
            'topn_genes',
            'Number of Example Genes per Locus',
            min=1,
            max=5,
            value=2,
            step=1.0,
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
                'min_contrib',
                'Min variance component:',
                value=0,
                min=0,
                max=1,
                step=0.01,
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
            'manhattan_trait',
            'Select Trait:',
            choices=list(PHENO_GUIDE),
            selected="Alzheimer's disease/dementia",
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
        <li><strong>Red</strong>: Loads onto 1st latent (E = enrichment score)</li>
        <li><strong>Orange</strong>: Loads onto 2nd latent</li>
        <li><strong>Green</strong>: Loads onto 3rd latent</li>
        <li><strong>Gray</strong>: Loads onto other latents</li>
    </ul>
    <p>Enrichment score: E = (g_sig/g_total) / (n_sig/n_total)</p>
    <p><strong>Reference:</strong> <a href="https://www.biorxiv.org/content/10.1101/2024.05.03.592285v2" target="_blank">Lazarev et al. bioRxiv 2024</a></p>
    """
    ),
    ui.output_plot('manhattan_plot', height='600px'),
    ui.output_ui('manhattan_info_ui'),
)
app_ui = ui.page_navbar(
    ui.nav_panel('Table', table),
    ui.nav_panel('Bar Plots', plots),
    ui.nav_panel('Manhattan Plots', manhattan_ui),
    title='GUI for GUIDE',
    id='main_navbar',
)


def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.plot
    def plot_variance_components_controller():
        interested_phenos = input.pheno_select()
        if isinstance(interested_phenos, str):
            interested_phenos = [interested_phenos]
        paths = [
            f'./all_phenos/phenotypes/guide/{get_safe_phe_name(p)}/squared_cosine_scores.tsv'
            for p in interested_phenos
        ]
        try:
            df = pd.read_csv(paths[0], sep='\t')
        except:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(
                0.5,
                0.5,
                'Data file not found',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
            ax.axis('off')
            return fig
        topn_var = input.n_var_components()
        fig, ax, latent_labels_local = plot_variance_components(
            df,
            ['squared_cosine_score'],
            [interested_phenos],
            topn=topn_var,
            bar_width=0.0001,
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
        interested_phenos = input.pheno_select()
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
        fig = pheno_contribs_to_lat(df, LATENT_LABELS, topn=input.n_phenotypes_show(), spacing=3)
        return fig

    @session.download(
        filename=lambda: f'guide-genes-{LATENT_LABELS[0].replace(" ", "_") if LATENT_LABELS else "none"}-{date.today().isoformat()}.csv'
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
                color: #0d6efd;
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
            latent_info = row['latent_info']
            html_parts.append(f"""
                <tr>
                    <td class="pheno-name">{pheno_name}</td>
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
                contrib = info['contribution']
                w_val = info['w_value']
                w_str = f'{w_val:.4f}' if not pd.isna(w_val) else 'N/A'
                html_parts.append(f"""
                            <tr>
                                <td class="latent-col">{latent}</td>
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
        if not input.manhattan_trait():
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
        selected_trait = input.manhattan_trait()
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
        if not input.manhattan_trait():
            return ui.markdown('**Please select a trait.**')
        selected_trait = input.manhattan_trait()
        df = prepare_manhattan_data(VAR_LABELS, CONTRIB_VAR_GUIDE, selected_trait, CONTRIB_GUIDE)
        if df is None:
            return ui.markdown('**Error: Could not load data for trait.**')
        top3_latent_labels = df.attrs.get('top3_latent_labels', [])
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

        for i, (lat_label, contrib, w_val) in enumerate(
            zip(top3_latent_labels, top3_contributions, top3_w_values), 1
        ):
            if not pd.isna(w_val):
                info_parts.append(
                    f'{i}. **{lat_label}** (variance component: {contrib:.4f}, -log10(w-value): {w_val:.4f})'
                )
            else:
                info_parts.append(f'{i}. **{lat_label}** (variance component: {contrib:.4f})')

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
        return ui.markdown('\n'.join(info_parts))

    @session.download(
        filename=lambda: f'manhattan-{input.manhattan_trait().replace("/", "_").replace(" ", "_") if input.manhattan_trait() else "data"}-{date.today().isoformat()}.csv'
    )
    async def download_manhattan_data():
        if not input.manhattan_trait():
            yield 'Error: No trait selected\n'
            return
        selected_trait = input.manhattan_trait()
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


app = App(app_ui, server, debug=True)
