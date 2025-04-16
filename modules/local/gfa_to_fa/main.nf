process GFA_TO_FA {
    label 'process_medium'

	 // needs to be called twice on the hifiasm output
	 // to allow mapping to unitigs

    input:
    tuple val(meta), path(gfa)

    output:
    tuple val(meta), path("*.fa{.gz,}"), emit: fasta

    when:
    task.ext.when == null || task.ext.when

    script:
    def args         = task.ext.args ?: ''
	 def compress     = true
    def prefix       = task.ext.prefix ?: "${meta.id}"
    def write_output = compress ? "| gzip > ${prefix}.fa.gzs" : "> ${prefix}.fa"
    """
	 awk '/^S/{print \">\"\$2;print \$3}' ${gfa} ${write_output}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        awk: \$( awk --version | head -n 1 | sed -e "s/,.*\$//g" )
    END_VERSIONS
    """
}
