#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// collection to start writing a main nextflow pipeline

include { HIC           } from './workflows/hic'
include { INPUT_CHECK   } from './subworkflows/local/input_check'
include { GFA_TO_GRAPH  } from './modules/local/gfa_to_graph'
include { GFA_TO_FA     } from './modules/local/gfa_to_fa'
include { ADD_HIC_EDGES } from './modules/local/descongelador'
include { HIFIASM       } from './modules/nf-core/hifiasm'


workflow GRAPHIC{
	take:
	samplesheet

	main:

	// parse input
	INPUT_CHECK( samplesheet )

	// TODO run fastqc on input Hifi reads?

	// run hifiasm to get unitigs
	HIFIASM(
		//INPUT_CHECK.out.reads.map { it -> [it[0], it[1].collectFile(), false }, // hifiasm should be able to handle multiple inputs
		INPUT_CHECK.out.reads.map { it -> [it[0], it[1], []] },
		[[], [], []],
		[[], [], []]
	)

	// start graph construction already, can run in parallel
	GFA_TO_GRAPH(HIFIASM.out.raw_unitigs)

	GFA_TO_FA(HIFIASM.out.raw_unitigs)

	ch_utigs = GFA_TO_FA.out.fasta

	// align hic reads to unitigs
	HIC(
		ch_utigs,
		INPUT_CHECK.out.hic_reads.map { it -> [it[0], it[1][0][0], it[1][0][1]] }
		// convert to [id, reads1, reads2]
	)

	// transform to graph structure, add hic edges
	ADD_HIC_EDGES(HIC.out.cool, GFA_TO_GRAPH.out.utg_graph, GFA_TO_GRAPH.out.utg_to_node)

	// emit as training dataset

	// deploy graphic model


	// feed back into hifiasm?

}

// init for main workflow
workflow {
	GRAPHIC(file(params.input))
}




///*
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    VALIDATE INPUTS
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//*/
//
//def summary_params = NfcoreSchema.paramsSummaryMap(workflow, params)
//
//// Validate input parameters
//WorkflowHic.initialise(params, log)
//
//// Check input path parameters to see if they exist
//def checkPathParamList = [ params.input ]
//checkPathParamList = [
//    params.input, params.multiqc_config,
//    params.fasta, params.bwt2_index
//]
//
//for (param in checkPathParamList) { if (param) { file(param, checkIfExists: true) } }
//
//// Check mandatory parameters
//if (params.input) { ch_input = file(params.input) } else { exit 1, 'Input samplesheet not specified!' }
//
////*****************************************
//// Digestion parameters
//if (params.digestion){
//  restriction_site = params.digestion ? params.digest[ params.digestion ].restriction_site ?: false : false
//  ch_restriction_site = Channel.value(restriction_site)
//  ligation_site = params.digestion ? params.digest[ params.digestion ].ligation_site ?: false : false
//  ch_ligation_site = Channel.value(ligation_site)
//}else if (params.restriction_site && params.ligation_site){
//  ch_restriction_site = Channel.value(params.restriction_site)
//  ch_ligation_site = Channel.value(params.ligation_site)
//}else if (params.dnase){
//  ch_restriction_site = Channel.empty()
//  ch_ligation_site = Channel.empty()
//}else{
//   exit 1, "Ligation motif not found. Please either use the `--digestion` parameters or specify the `--restriction_site` and `--ligation_site`. For DNase Hi-C, please use '--dnase' option"
//}
//
//
////
//// SUBWORKFLOW: Prepare genome annotation
////
//PREPARE_GENOME(
// ch_fasta,
// ch_restriction_site
//)
//ch_versions = ch_versions.mix(PREPARE_GENOME.out.versions)
//
////
//// MODULE: Run FastQC
////
//FASTQC (
// INPUT_CHECK.out.reads
//)
//ch_versions = ch_versions.mix(FASTQC.out.versions)
//
////
//// SUB-WORFLOW: HiC-Pro
////
//INPUT_CHECK.out.reads.view()
//HICPRO (
// INPUT_CHECK.out.reads,
// PREPARE_GENOME.out.index,
// PREPARE_GENOME.out.res_frag,
// PREPARE_GENOME.out.chromosome_size,
// ch_ligation_site
//)
//ch_versions = ch_versions.mix(HICPRO.out.versions)
//
////
//// SUB-WORKFLOW: COOLER
////
//COOLER (
// HICPRO.out.pairs,
// PREPARE_GENOME.out.chromosome_size
//)
//ch_versions = ch_versions.mix(COOLER.out.versions)
