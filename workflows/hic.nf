/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    VALIDATE INPUTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

nextflow.enable.dsl = 2

// Check input path parameters to see if they exist
def checkPathParamList = [ params.input ]
checkPathParamList = [
    params.input, params.multiqc_config
]

for (param in checkPathParamList) { if (param) { file(param, checkIfExists: true) } }

// Check mandatory parameters
if (params.input) { ch_input = file(params.input) } else { exit 1, 'Input samplesheet not specified!' }

//*****************************************
// Digestion parameters
if (params.digestion){
  restriction_site = params.digestion ? params.digest[ params.digestion ].restriction_site ?: false : false
  ch_restriction_site = Channel.value(restriction_site)
  ligation_site = params.digestion ? params.digest[ params.digestion ].ligation_site ?: false : false
  ch_ligation_site = Channel.value(ligation_site)
}else if (params.restriction_site && params.ligation_site){
  ch_restriction_site = Channel.value(params.restriction_site)
  ch_ligation_site = Channel.value(params.ligation_site)
}else if (params.dnase){
  ch_restriction_site = Channel.empty()
  ch_ligation_site = Channel.empty()
}else{
   exit 1, "Ligation motif not found. Please either use the `--digestion` parameters or specify the `--restriction_site` and `--ligation_site`. For DNase Hi-C, please use '--dnase' option"
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CONFIG FILES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

ch_multiqc_config          = Channel.fromPath("$projectDir/assets/multiqc_config.yml", checkIfExists: true)
ch_multiqc_custom_config   = params.multiqc_config ? Channel.fromPath( params.multiqc_config, checkIfExists: true ) : Channel.empty()
ch_multiqc_logo            = params.multiqc_logo   ? Channel.fromPath( params.multiqc_logo, checkIfExists: true ) : Channel.empty()
ch_multiqc_custom_methods_description = params.multiqc_methods_description ? file(params.multiqc_methods_description, checkIfExists: true) : file("$projectDir/assets/methods_description_template.yml", checkIfExists: true)

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT LOCAL MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// MODULE: Local to the pipeline
//
//include { HIC_PLOT_DIST_VS_COUNTS } from '../modules/local/hicexplorer/hicPlotDistVsCounts' 
include { MULTIQC } from '../modules/local/multiqc'

//
// SUBWORKFLOW: Consisting of a mix of local and nf-core/modules
//
include { PREPARE_GENOME } from '../subworkflows/local/prepare_genome'
include { HICPRO } from '../subworkflows/local/hicpro'
include { COOLER } from '../subworkflows/local/cooler'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// MODULE: Installed directly from nf-core/modules
//
include { FASTQC                      } from '../modules/nf-core/fastqc/main'
include { CUSTOM_DUMPSOFTWAREVERSIONS } from '../modules/nf-core/custom/dumpsoftwareversions/main'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

// Info required for completion email and summary
def multiqc_report = []

workflow HIC {
  take:
  ch_fasta // filepath
  ch_hic_reads // channel emitting pairs of files

  main:

  ch_versions = Channel.empty()

  //
  // SUBWORKFLOW: Prepare genome annotation
  //
  PREPARE_GENOME(
    ch_fasta,
    ch_restriction_site
  )
  ch_versions = ch_versions.mix(PREPARE_GENOME.out.versions)

  //
  // MODULE: Run FastQC
  //
  FASTQC (
    ch_hic_reads
  )
  ch_versions = ch_versions.mix(FASTQC.out.versions)

  //
  // SUB-WORFLOW: HiC-Pro
  //
  INPUT_CHECK.out.reads.view()
  HICPRO (
    INPUT_CHECK.out.reads,
    PREPARE_GENOME.out.index,
    PREPARE_GENOME.out.res_frag,
    PREPARE_GENOME.out.chromosome_size,
    ch_ligation_site
  )
  ch_versions = ch_versions.mix(HICPRO.out.versions)

  //
  // SUB-WORKFLOW: COOLER
  //
  COOLER (
    HICPRO.out.pairs,
    PREPARE_GENOME.out.chromosome_size
  )
  ch_versions = ch_versions.mix(COOLER.out.versions)

  //
  // MODULE: HICEXPLORER/HIC_PLOT_DIST_VS_COUNTS
  //
  // probably won't need the plotting
  //if (!params.skip_dist_decay){
  //  COOLER.out.cool
  //    .combine(ch_ddecay_res)
  //    .filter{ it[0].resolution == it[2] }
  //    .map { it -> [it[0], it[1]]}
  //    .set{ ch_distdecay }

  //  HIC_PLOT_DIST_VS_COUNTS(
  //    ch_distdecay
  //  )
  //  ch_versions = ch_versions.mix(HIC_PLOT_DIST_VS_COUNTS.out.versions)
  //}

  //
  // SOFTWARE VERSION
  //
  CUSTOM_DUMPSOFTWAREVERSIONS(
    ch_versions.unique().collectFile(name: 'collated_versions.yml')
  )

  //
  // MODULE: MultiQC
  //

  summary_params = NfcoreSchema.paramsSummaryMap(workflow, params)
  workflow_summary    = WorkflowHic.paramsSummaryMultiqc(workflow, summary_params)
  ch_workflow_summary = Channel.value(workflow_summary)

  ch_multiqc_files = Channel.empty()
  ch_multiqc_files = ch_multiqc_files.mix(ch_multiqc_config)
  ch_multiqc_files = ch_multiqc_files.mix(ch_multiqc_custom_config.collect().ifEmpty([]))
  ch_multiqc_files = ch_multiqc_files.mix(ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
  ch_multiqc_files = ch_multiqc_files.mix(FASTQC.out.zip.map{it->it[1]})
  ch_multiqc_files = ch_multiqc_files.mix(HICPRO.out.mqc)

  MULTIQC (
    ch_multiqc_config,
    ch_multiqc_custom_config.collect().ifEmpty([]),
    ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'),
    FASTQC.out.zip.map{it->it[1]},
    HICPRO.out.mqc.collect()
  )
  multiqc_report = MULTIQC.out.report.toList()

  emit:
  versions = ch_versions
  cool = COOL.out.cool
  multiqc_report = multiqc_report
}
