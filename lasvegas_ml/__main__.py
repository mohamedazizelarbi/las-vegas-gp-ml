"""Command line for the Las Vegas Grand Prix forecast project."""

import argparse
from importlib import import_module


COMMANDS = {
    'audit': ('simulation.audit', 'Compare locked forecasts with the FIA final classification'),
    'simulate': ('simulation.model', 'Refit the pace and retirement models and repeat the rolling backtest'),
    'train-winner': ('v2.training', 'Refit and backtest the v2 race-winner models'),
    'compare-features': ('v2.compare_features', 'Test long-run pace, recent race pace, and DNF rate separately'),
    'test-dnf-calibration': ('simulation.dnf_calibration', 'Replay the retirement calibration experiment'),
    'download': ('v2.download_data', 'Download the original study inputs from OpenF1'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text, description=help_text)
    args = parser.parse_args()
    module_name, _ = COMMANDS[args.command]
    import_module(f'.{module_name}', __package__).main()


if __name__ == '__main__':
    main()
