from argparse import ArgumentParser
from figure3 import draw_figure3
from figure4 import draw_figure4
from figure5 import draw_figure5

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='Specifies if the figures are generated from saved files.', action='store_true')
    args = vars(parser.parse_args())
    draw_figure3(show=False, use_cached_file=args['file'])
    draw_figure4(show=False, use_cached_file=args['file'])
    draw_figure5(show=False, use_cached_file=args['file'])
