"""
Author: @arcsi1989
"""
import click

from src.cli import src_cli


@src_cli.command(help_priority=2)
@click.help_option("-h")
@click.option("-o", "--output_folder",
              type=str,
              help="Provide a folder where the output should be placed")
def inference(output_folder: str):
    print('LOG | Pipeline for predicting movie views using a pretrained model')
