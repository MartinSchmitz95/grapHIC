//
// Check input samplesheet and get read and hic read channels
//

include { SAMPLESHEET_CHECK } from '../../modules/local/samplesheet_check'

workflow INPUT_CHECK {
take:
	samplesheet // file: /path/to/samplesheet.csv

		main:
		if (params.split_fastq){

			SAMPLESHEET_CHECK ( samplesheet )
				.csv
				.splitCsv ( header:true, sep:',' )
				.map { it -> create_fastq_channels(it) }
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
			SAMPLESHEET_CHECK ( samplesheet )
				.csv
				.splitCsv ( header:true, sep:',' )
				.map { it -> create_fastq_channels(it) }
				// group by sample name in meta
				.map { it -> [it[0], [it[1], [it[2], it[3]]]] }
				.groupTuple(by: [0])
				// do separately for normal & hic reads
				//.flatMap { it -> setMetaChunk(it) } // puts technical replicates into meta, disaggregates grouped tuples
				.multiMap {
					it ->
						reads: [it[0], it[1]]
						hic_reads: [it[0], it[2]]//[it[2], it[3]]]
				}
				.set { merged_in }
		}

	// export as output
	merged_in.reads.flatMap { it -> 
			def map = []
			it[1].eachWithIndex() { file,i ->
				meta = row[0].clone()
				meta.chunk = i
				map += [meta, file]
		}.set{ reads }

	merged_in.hic_reads.flatMap { it -> 
			def map = []
			it[1].eachWithIndex() { tup,i ->
				meta = row[0].clone()
				meta.chunk = i
				map += [meta, tup]
		}.set{ hic_reads }

	reads.view()
	hic_reads.view()

emit:
	reads // channel: [ val(meta), [ reads ] ]
	hic_reads // channel: [ val(meta), [ [reads_1, reads_2] ] ]
}

// Function to get list of [ meta, [ reads, hic1, hic2 ] ]
def create_fastq_channels(LinkedHashMap row) {
	def meta = [:]
	meta.id = row.id
	meta.single_end = false

	def reads = Channel.of()
	def hic_reads1 = Channel.of()
	def hic_reads2 = Channel.of()

	def error = false // store error state and die at the end, so all errors get reported at once

	// if a file is given, check it exists
	if (row.reads) {
		reads = file(row.reads)
		if (!reads.exists()){
			print("ERROR: Please check input samplesheet -> Reads file does not exist!\n${row.reads}")
			error = true
		}
	}

	// not specifying any files is also valid
	if ( (row.hic_reads1 as Boolean) ^ (row.hic_reads2 as Boolean) ) {
		print("ERROR: HiC files need to be specified in pairs!\n${row.hic_reads1} or ${row.hic_reads2} does not have a paired file!")
		error = true
	}

	if (row.hic_reads1) {
		hic_reads1 = file(row.hic_reads1)
		if (!hic_reads1.exists()){
			print("ERROR: Please check input samplesheet -> Reads file does not exist!\n${row.hic_reads1}")
			error = true
		}
	}

	if (row.hic_reads2) {
		hic_reads2 = file(row.hic_reads2)
		if (!hic_reads2.exists()){
			print("ERROR: Please check input samplesheet -> Reads file does not exist!\n${row.hic_reads2}")
			error = true
		}
	}

	if (error){
		exit 1, "ERROR: Samplesheet invalid (see above errors)"
	}

	array = [ meta, reads, hic_reads1, hic_reads2 ]
	return array
}
