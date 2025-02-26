import matplotlib.pyplot as plt
import matplotlib as mpl


def setup_matplotlib_style(big_fonts=False, tex_fonts=True, medium_fonts=False, font_size = None):
    """Set up matplotlib style with larger fonts, without LaTeX dependencies"""
    plt.style.use('default')
    
    plt.rcParams.update({
        # Use serif fonts
        'font.family': 'serif',

        # Set minimum font sizes
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,

        # Use mathtext instead of LaTeX
        'text.usetex': False,
        'mathtext.default': 'regular',
        'axes.formatter.use_mathtext': True,
        
        # Increase figure size for better visibility
        'figure.figsize': [12, 8],
        'figure.dpi': 100,

        # Other styling
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    if big_fonts:
        plt.rcParams.update({
            'font.size': 32,
            'axes.labelsize': 32,
            'axes.titlesize': 32,
            'xtick.labelsize': 32,
            'ytick.labelsize': 32,
            'legend.fontsize': 32,
        })
        
    if medium_fonts:
        plt.rcParams.update({
            'font.size': 28,
            'axes.labelsize': 28,
            'axes.titlesize': 28,
            'xtick.labelsize': 28,
            'ytick.labelsize': 28,
            'legend.fontsize': 28,
        })
        
    if tex_fonts:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}'
        })
        
    if font_size is not None:
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
        })