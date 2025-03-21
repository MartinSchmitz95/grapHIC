process MAKE_HIC_EDGES {
    tag "$meta.id"
    label 'process_single'

    conda "${moduleDir}/env_desc.yml"

    input:
    tuple val(meta), path(gfa)
    tuple path(gfa)

    output:
    tuple val(meta), path("*.nx.pkl"), emit: utg_graph
    tuple val(meta), path("*.nx.pkl"), emit: utg_to_node
    tuple val(meta), path("*.nx.pkl"), emit: utg_to_reads
    path ("versions.yml")            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python3 ${moduleDir}/gfa_to_graph.py -i ${gfa}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gfa_to_graph: 1.0
    END_VERSIONS
    """
}
