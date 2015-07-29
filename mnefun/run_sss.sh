#!/bin/bash -ef
# Copyright (c) 2014, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#
# This script is designed to facilitate running of SSS on a remote
# machine.
#

#if ! command -v maxfilter > /dev/null; then
#    echo "ERROR: maxfilter not found, consider adding '/neuro/bin/util' to PATH"
#    exit 1
#fi

PATH=$PATH:/neuro/bin/util

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ERM_RUNS=""
HEAD_TRANS="median"
ST_DUR="60"
FORMAT="float"
EXTRA_ARGS=""

ASSERT_ISFILE () {
    if [ ! -f "$1" ] ; then
        echo "ERROR: File not found:"
        echo "  $1"
        exit 1
    fi
}

GET_PRE_POST_FIX () {
    local __test=`echo ${3} | sed 's/\(.*\)_raw\(-\?[0-9]*\?\).fif//'`
    if [ "${__test}" != "" ]; then
        echo "ERROR: Improperly named raw file $3"
        exit 1
    fi;
    local __resultvar=$1
    local myresult=`echo ${3} | sed 's/\(.*\)_raw\(-\?[0-9]*\?\).fif/\1/'`
    eval $__resultvar="'$myresult'"
    local __resultvar=$2
    local myresult=`echo ${3} | sed 's/\(.*\)_raw\(-\?[0-9]*\?\).fif/\2/'`
    eval $__resultvar="'$myresult'"
}

#################################################### Read the options ##
TEMP=`getopt -o s:f:e:t: --long subject:,files:,erm:,trans:,format:,st:,args: -n 'run_sss.sh' -- "$@"`
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -s|--subject)
            case "$2" in
                "") shift 2 ;;
                *) SUBJECT=$2 ; shift 2 ;;
            esac ;;
        -f|--files)
            case "$2" in
                "") shift 2 ;;
                *) FILES=$2 ; shift 2 ;;
            esac ;;
        -e|--erm)
            case "$2" in
                "") shift 2 ;;
                *) ERM_FILES=$2 ; shift 2 ;;
            esac ;;
        --format)
            case "$2" in
                "") shift 2 ;;
                *) FORMAT=$2 ; shift 2 ;;
            esac ;;
        -t|--trans)
            case "$2" in
                "") shift 2 ;;
                *) HEAD_TRANS=$2 ; shift 2 ;;
            esac ;;
        --st)
            case "$2" in
                "") shift 2 ;;
                *) ST_DUR=$2 ; shift 2 ;;
            esac ;;
        --args)
            case "$2" in
                "") shift 2 ;;
                *) EXTRA_ARGS=$2 ; shift 2 ;;
            esac ;;
        --) shift ; break ;;
        *) echo "ERROR: Internal error!" ; exit 1 ;;
    esac
done

############################################################# Run SSS ##
echo "Running subject '${SUBJECT}' in '${ROOT_DIR}'"

RAW_DIR="${ROOT_DIR}/${SUBJECT}/raw_fif/"
SSS_DIR="${ROOT_DIR}/${SUBJECT}/sss_fif/"
LOG_DIR="${ROOT_DIR}/${SUBJECT}/sss_log/"
mkdir -p "${SSS_DIR}"
mkdir -p "${LOG_DIR}"

# Forcing bad channels
BADFILE="${RAW_DIR}${SUBJECT}_prebad.txt"
if [ -f ${BADFILE} ]; then
    EXTRABADS=`cat $BADFILE`
    echo "• Forcing bad channels (${EXTRABADS})"
else
    EXTRABADS=""
    echo "• Not forcing extra bad channels"
fi

# Head position translation
IFS=':' read -ra ADDR <<< "$FILES"
case "${HEAD_TRANS}" in 
    "first")
        HEAD_TRANS="${RAW_DIR}${ADDR[0]}"
        echo "• Translating to first run head position"
        ;;
    "median")
        HEAD_TRANS="${RAW_DIR}${SUBJECT}_median_pos.fif"
        echo "• Translating to the median head position"
        ;;
    "default")
        HEAD_TRANS="default"
        echo "• Translating to the default head position"
        ;;
    *)
        HEAD_TRANS="${ROOT_DIR}/${HEAD_TRANS}"
        echo "• Translating to common head position: ${HEAD_TRANS}"
esac
if [ ${HEAD_TRANS} != "default" ]; then
    ASSERT_ISFILE "${HEAD_TRANS}"
fi;
# Head center
CENTER_FILE="${RAW_DIR}${SUBJECT}_center.txt"
ASSERT_ISFILE "${CENTER_FILE}"
RUN_CENTER=`cat ${CENTER_FILE}`
echo "• Using head center ${RUN_CENTER}"

# Extra arguments
if [[ ! -z ${EXTRA_ARGS} ]]; then
    echo "• Using extra arguments '${EXTRA_ARGS}'"
fi;

# Standard SSS parameters
ERM_FRAME="-frame device -origin 0 13 -6"
RUN_FRAME="-frame head -origin ${RUN_CENTER}"

BAD_PARAMS="-autobad 20 -force -v -format short"
ST_PARAMS="-in 8 -out 3 -regularize in -st ${ST_DUR}"
MC_PARAMS="-trans ${HEAD_TRANS} -hpicons -movecomp inter -format ${FORMAT}"

# Run processing
for FILE in "${ADDR[@]}"; do
    echo ""
    echo "Processing run: ${FILE}"
    RAW_FILE="${RAW_DIR}${FILE}"
    GET_PRE_POST_FIX PREFIX POSTFIX ${FILE}
    THIS_POS="${RAW_DIR}${HEAD_TRANS}"
    ASSERT_ISFILE "${RAW_FILE}"

    # Bad channels
    echo "• Auto-detecting bad channels"
    OUT_FILE="${SSS_DIR}${PREFIX}_raw_sss_badch${POSTFIX}.fif"
    BADCH=`maxfilter -f ${RAW_FILE} -o ${OUT_FILE} ${RUN_FRAME} ${BAD_PARAMS} \
           2>/dev/null \
           | tee ${LOG_DIR}${PREFIX}_1_bads${POSTFIX}.txt \
           | sed -n  '/Static bad channels/p' \
           | cut -f 5- -d ' ' | uniq | xargs printf "%04d "`
    rm -f ${OUT_FILE}
    echo "• Found: ${BADCH}"
    if [ "${BADCH}" == "0000 " ]; then BADCH=""; fi
    if [ "${BADCH}" != "" ] || [ "${EXTRABADS}" != "" ]; then
        BADCH="-bad ${BADCH}${EXTRABADS}"
    fi
    BADCH="-autobad off ${BADCH}"

    MC_LOG="-hp ${LOG_DIR}${PREFIX}_hp${POSTFIX}.txt"
    SSS_FILE="${SSS_DIR}${PREFIX}_raw_sss${POSTFIX}.fif"

    # Singleshot SSS (+st +mc +cs transform +hpi file) procedure
    # Show user arguments, minus logging / file args
    ARGS="${RUN_FRAME} ${BADCH} ${ST_PARAMS} ${MC_PARAMS} ${EXTRA_ARGS} -force -v"
    echo "• Running maxfilter ${ARGS}"
    maxfilter -f ${RAW_FILE} -o ${SSS_FILE} ${MC_LOG} ${ARGS} &> ${LOG_DIR}${PREFIX}_sss${POSTFIX}.txt
done

# ERM processing
IFS=':' read -ra ADDR <<< "$ERM_FILES"
for FILE in "${ADDR[@]}"; do
    echo ""
    echo "Processing empty room: ${FILE}"
    RAW_FILE="${RAW_DIR}${FILE}"
    GET_PRE_POST_FIX PREFIX POSTFIX ${FILE}
    ASSERT_ISFILE "${RAW_FILE}"

    # Bad channels
    echo "• Auto-detecting bad channels"
    OUT_FILE="${SSS_DIR}${PREFIX}_raw_sss_badch${POSTFIX}.fif"
    BADCH=`maxfilter -f ${RAW_FILE} -o ${OUT_FILE} ${ERM_FRAME} ${BAD_PARAMS} \
          2>/dev/null \
          | tee ${LOG_DIR}${PREFIX}_1_bads${POSTFIX}.txt \
          | sed -n  '/Static bad channels/p' \
          | cut -f 5- -d ' ' | uniq | xargs printf "%04d "`
    rm -f ${OUT_FILE}
    echo "• Found: ${BADCH}"
    if [ "${BADCH}" == "0000 " ]; then BADCH=""; fi
    if [ "${BADCH}" != "" ] || [ "${EXTRABADS}" != "" ]; then
        BADCH="-bad ${BADCH}${EXTRABADS}"
    fi
    BADCH="-autobad off ${BADCH}"
    SSS_FILE="${SSS_DIR}${PREFIX}_raw_sss${POSTFIX}.fif"

    # singleshot SSS procedure (no MC)
    ARGS="${ERM_FRAME} ${BADCH} ${ST_PARAMS} ${EXTRA_ARGS}"
    echo "• Running maxfilter ${ARGS}"
    maxfilter -f ${RAW_FILE} -o ${SSS_FILE} ${ARGS} -force -v &> ${LOG_DIR}${PREFIX}_sss${POSTFIX}.txt
done

echo ""
echo "SSS Successfully completed"
