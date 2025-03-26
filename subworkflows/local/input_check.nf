//
// Check input samplesheet and get read and hic read channels
//

include { SAMPLESHEET_CHECK } from '../../modules/local/samplesheet_check'

workflow INPUT_CHECK {
    take:
    samplesheet // file: /path/to/samplesheet.csv

    main:
    if (params.split_fastq){

      SAMPLESHEET_CHECK ( samplesheet ).out.csv
			.splitCsv ( header:true, sep:',' )
			.map { create_fastq_channels(it) }
			.splitFastq( by: params.fastq_chunks_size, pe:true, file: true, compress:true)
			// group by sample name in meta
			.map { it -> [it[0], [it[1], [it[2], it[3]]]]}
			.groupTuple(by: [0])
			.flatMap { it -> setMetaChunk(it) } // puts technical replicates into meta, disaggregates grouped tuples
			.multiMap {
				it ->
					hic_reads: [it[0], [it[2], it[3]]]
					reads: [it[0], it[1]]
				}
			.set { merged_in }

	// this step used to rely on undefined behaviour, I think
	//.flatMap { it -> setMetaChunk(it) }
	//.collate(2)
	//.set { hic_reads }


    } else {
      SAMPLESHEET_CHECK ( samplesheet ).out.csv
			.splitCsv ( header:true, sep:',' )
			.map { create_fastq_channels(it) }
			// group by sample name in meta
			.map { it -> [it[0], [it[1], [it[2], it[3]]]]}
			.groupTuple(by: [0])
			.flatMap { it -> setMetaChunk(it) } // puts technical replicates into meta, disaggregates grouped tuples
			.multiMap {
				it ->
					hic_reads: [it[0], [it[2], it[3]]]
					reads: [it[0], it[1]]
				}
			.set { merged_in }
   }

	// export as output
	merged_in.hic_reads.set{ hic_reads }
	merged_in.reads.set{ reads }

    emit:
    reads // channel: [ val(meta), [ reads ] ]
    hic_reads // channel: [ val(meta), [ [reads_1, reads_2] ] ]
}

// Function to get list of [ meta, [ fastq_1, fastq_2 ] ]
def create_fastq_channels(LinkedHashMap row) {
  def meta = [:]
  meta.id = row.sample
  meta.single_end = false

  def array = []
  if (!file(row.fastq_1).exists() & file(row.fastq_2).exists()) {
    exit 1, "ERROR: Please check input samplesheet -> Read 1 FastQ file does not exist!\n${row.fastq_1}"
  }
  if (!file(row.fastq_2).exists() & file(row.fastq_1).exists()) {
    exit 1, "ERROR: Please check input samplesheet -> Read 2 FastQ file does not exist!\n${row.fastq_2}"
  }
  // in case there are only regular reads given as input
  if (!file(row.fastq_1).exists() & !file(row.fastq_2).exists()) {
	  return []
  }
  array = [ meta, file(row.reads), file(row.fastq_1), file(row.fastq_2) ]
  return array
}

// Set the meta.chunk value in case of technical replicates
def setMetaChunk(row){
  def map = []
  row[1].eachWithIndex() { file,i ->
    meta = row[0].clone()
    meta.chunk = i
    map += [meta, file]
  }
  return map
}
