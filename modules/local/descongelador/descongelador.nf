process MAKE_HIC_EDGES {
    tag "$meta.id"
    label 'process_single'

    conda "${moduleDir}/env_desc.yml"

    input:
    tuple val(meta), path(hic_contacts)
    tuple path(utg_dict)

    output:
    tuple val(meta), path("*.nx.pkl"), emit: graph
    path ("versions.yml")            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python3 ${moduleDir}/descongelador.py -i ${hic_contacts} -d ${utg_dict} -o contacts.nx.pkl

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        descongelador: 1.0
    END_VERSIONS
    """
}
