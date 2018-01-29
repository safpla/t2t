# coding=utf-8
# copyright 2017 Xu Haowen
#
# email: haowen.will.xu@gmail.com

import os, sys
import codecs
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)

class LabelFormat(object):
    '''
    Available label formats
    '''
    IO = 'IO'
    IOBES = 'IOBES'
    IOB = 'IOB'
    IOBE = 'IOBE'

def _io2iobe(line, io_dict=None, iobe_dict=None):
    if io_dict is None:
        io_dict = {'I':'I', 'O':'O'}
    if iobe_dict is None:
        iobe_dict = {'I':'I', 'O':'O', 'B':'B', 'E':'E'}
    iobe_line = []
    start = 0
    len_line = len(line)
    while start < len_line:
        if line[start] == io_dict['O']:
            iobe_line.append(iobe_dict['O'])
            start += 1
        else:
            stop = start + 1
            while stop < len_line and line[stop] == io_dict['I']:
                stop += 1
            entity = [iobe_dict['I']] * (stop - start)
            entity[-1] = iobe_dict['E']
            entity[0] = iobe_dict['B']
            iobe_line.extend(entity)
            start = stop
    return iobe_line

def _iob2iobe(line, iob_dict=None, iobe_dict=None):
    if iob_dict is None:
        iob_dict = {'I':'I', 'O':'O', 'B':'B'}
    if iobe_dict is None:
        iobe_dict = {'I':'I', 'O':'O', 'B':'B', 'E':'E'}

    iobe_line = []
    start = 0
    len_line = len(line)
    while start < len_line:
        if line[start] == iob_dict['O']:
            iobe_line.append(iobe_dict['O'])
            start += 1
        else:
            stop = start + 1
            while stop < len_line and line[stop] == iob_dict['I']:
                stop += 1
            entity = [iobe_dict['I']] * (stop - start)
            entity[-1] = iobe_dict['E']
            entity[0] = iobe_dict['B']
            iobe_line.extend(entity)
            start = stop
    return iobe_line

def _iobe2iobe(line, iobe_dict1=None, iobe_dict2=None):
    if iobe_dict1 is None:
        iobe_dict1 = {'I':'I', 'O':'O', 'B':'B', 'E':'E'}
    if iobe_dict2 is None:
        iobe_dict2 = {'I':'I', 'O':'O', 'B':'B', 'E':'E'}
    replace_dict = {}
    for key1, value1 in iobe_dict1.items():
        replace_dict[value1] = iobe_dict2[key1]
    new_line = [replace_dict[char] for char in line]
    return new_line

def label_translate_line(source, source_format, target_format, source_dict=None,
                         target_dict=None):
    '''
    see label_translate
    '''
    translate_fn = TRANSLATE_FNS[source_format, target_format]
    return translate_fn(source, source_dict, target_dict)

def label_translate(source, target, source_format, target_format, source_dict=None,
                    target_dict=None, input_type='file', separator=' '):
    '''
    translate one label format to another.
    optional formats: LabelFormat, [IO, IOB, IOBE, IOBES].
    optional input_type: string, [file, line].
    in case the origin labels have different representation, use source_dict
    and target_dict to specify them.
    Args:
        source: file name or label sequence(string or list), depands on input_type
        target: file name or null, depands on input_type
        source_format: LabelFormat
        target_format: LabelFormat
        source_dict: dictionary {'1':'I', '0':'O'}
        target_dict: dictionary
        input_type: string
        separator: used to separate source if source is string
    Returns:
        null if input_type is file or translated label sequence(list)
    '''
    if input_type == 'line':
        if isinstance(source, str):
            source = source.strip().split(separator)
        return(label_translate_line(source, source_format, target_format,
                                    source_dict, target_dict))
    else:
        source_stream = codecs.open(source, 'r', 'utf-8')
        target_stream = codecs.open(target, 'w', 'utf-8')
        for line in source_stream.readlines():
            line = line.strip().split(separator)
            line_target = label_translate_line(line, source_format, target_format,
                                               source_dict, target_dict)
            target_stream.write(separator.join(line_target) + '\n')
        source_stream.close()
        target_stream.close()


TRANSLATE_FNS = {
    (LabelFormat.IO, LabelFormat.IOBE): _io2iobe,
    (LabelFormat.IOB, LabelFormat.IOBE): _iob2iobe,
    (LabelFormat.IOBE, LabelFormat.IOBE): _iobe2iobe,
}
if __name__ == '__main__':
    source = 'O O I-PER I-PER O O O I-PER I-PER B-PER E-PER I-PER O O'
    iobe_dict = {'O':'O', 'I':'I-PER', 'B':'B-PER', 'E':'E-PER'}
    print(label_translate(source, '', LabelFormat.IOBE, LabelFormat.IOBE,
                          source_dict=iobe_dict, input_type='line', separator=' '))
