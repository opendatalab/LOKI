from itertools import islice


def create_iterator(raw_iterator, rank, world_size, limit=None):
    """
    Method for creating a (potentially) sliced and limited
    iterator from a raw document iterator. Used for splitting data
    among ranks in multigpu setting or only pulling a sample of documents
    """
    return islice(raw_iterator, rank, limit, world_size)


def make_table(result_dict):
    """Generate table of results."""
    """
    for subset, subset_result in evaluation_result.items():
        printable_results[subset] = {
            "num": int(subset_result["num_example"]),
            "acc": round(subset_result["acc"], 5),
        }
    """
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    all_headers = [
        "Type",
        "Num",
        "Accuracy"
    ]

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    # Set column alignments for LaTeX
    latex_writer.column_alignments = ["center"] * len(all_headers)

    # Set padding for LaTeX columns (this will add space between columns)
    latex_writer.column_format = " ".join(["|c"] * len(all_headers)) + "|"

    values = []

    for subset, subset_result in result_dict.items():
        values.append([subset, subset_result['num'], subset_result['accuracy']])
            
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # Print LaTeX table to see how it looks
    # print(latex_writer.dumps())

    # Return Markdown table (note: column width and text alignment may not be supported)
    return md_writer.dumps()