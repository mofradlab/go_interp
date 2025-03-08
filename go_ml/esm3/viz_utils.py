import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
import bokeh as bh
import bokeh.plotting
# bokeh.io.output_notebook()


from esm.utils.constants.esm3 import SEQUENCE_VOCAB
esm_alphabet = SEQUENCE_VOCAB[4:24]
# ALPHABET = "AFILVMWYDEKRHNQSTGPC"
# ALPHABET_map = [esm_alphabet.index(a) for a in ALPHABET]
def pssm_to_dataframe(pssm, esm_alphabet):
    sequence_length = pssm.shape[0]
    idx = [str(i) for i in np.arange(1, sequence_length + 1)]
    df = pd.DataFrame(pssm, index=idx, columns=list(esm_alphabet))
    df = df.stack().reset_index()
    df.columns = ['Position', 'Amino Acid', 'Probability']
    return df

def mat_to_dataframe(mat, row_labels):
    sequence_length = mat.shape[0]
    idx = [str(i) for i in np.arange(1, sequence_length + 1)]
    df = pd.DataFrame(mat, index=idx, columns=list(esm_alphabet))
    df = df.stack().reset_index()
    df.columns = ['Position', 'Amino Acid', 'Probability']
    return df

def get_pssm_display(sequence, logits):
    pssm_df = pssm_to_dataframe(softmax(logits, axis=-1), esm_alphabet)
    num_colors = 256  # You can adjust this number
    palette = bh.palettes.viridis(256)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    p = bh.plotting.figure(title="CONSERVATION",
            x_range=[str(x) for x in range(1,len(sequence)+1)],
            y_range=list(esm_alphabet)[::-1],
            width=900, height=400,
            tools=TOOLS, toolbar_location='below',
            tooltips=[('Position', '@Position'), ('Amino Acid', '@{Amino Acid}'), ('Probability', '@Probability')])

    r = p.rect(x="Position", y="Amino Acid", width=1, height=1, source=pssm_df,
            fill_color=bh.transform.linear_cmap('Probability', palette, low=0, high=1),
            line_color=None)
    p.xaxis.visible = False  # Hide the x-axis
    return p

def get_score_display(sequence, scores):
    pssm_df = pssm_to_dataframe(scores, esm_alphabet)
    num_colors = 256  # You can adjust this number
    palette = bh.palettes.viridis(256)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    p = bh.plotting.figure(title="CONSERVATION",
            x_range=[str(x) for x in range(1,len(sequence)+1)],
            y_range=list(esm_alphabet)[::-1],
            width=900, height=400,
            tools=TOOLS, toolbar_location='below',
            tooltips=[('Position', '@Position'), ('Amino Acid', '@{Amino Acid}'), ('Score', '@Probability')])

    r = p.rect(x="Position", y="Amino Acid", width=1, height=1, source=pssm_df,
            fill_color=bh.transform.linear_cmap('Probability', palette, low=scores.min(), high=scores.max()),
            line_color=None)
    p.xaxis.visible = False  # Hide the x-axis
    return p
