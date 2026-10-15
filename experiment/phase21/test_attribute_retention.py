import unittest
import numpy as np
from experiment.phase21.attribute_retention_core import parse_fields, pack_catalog


class TinyTokenizer:
    def __init__(self): self.vocab={};self.inverse={}
    def __call__(self,text,add_special_tokens=True,**kwargs):
        values=[]
        for token in text.split():
            if token not in self.vocab:
                i=len(self.vocab)+10;self.vocab[token]=i;self.inverse[i]=token
            values.append(self.vocab[token])
        return {'input_ids':values+([1] if add_special_tokens else [])}
    def decode(self,ids,**kwargs):return ' '.join(self.inverse.get(int(i),'') for i in ids if i>1)


class PackingBoundaryTests(unittest.TestCase):
    def catalog(self):
        lexical=['']+[f'lex{i}' for i in range(1,8)]
        passages=['']
        for i in range(1,8):
            description='na' if i==7 else ' '.join([f'desc{i}']*300)
            passages.append(f'item: lex{i}; similar items: a, b, c, d, e; title: title{i}; categories: category; brand: brand; description: {description}; price: 2')
        return dict(items=['']+[f'item{i}' for i in range(1,8)],lexical_ids=lexical,passages=passages)
    def test_field_parser_keeps_semicolons_inside_description(self):
        fields=parse_fields('item: x; description: useful; more details; price: 2')
        self.assertEqual(fields['description'],'useful; more details')
    def test_placebo_preserves_every_other_token_and_length(self):
        catalog=self.catalog();tok=TinyTokenizer()
        banks,lengths,donors,audit=pack_catalog(catalog,tok,'Toys')
        np.testing.assert_array_equal(lengths['retained'],lengths['shuffled'])
        for i in range(1,8):
            a,b=audit['description_offsets'][str(i)]
            np.testing.assert_array_equal(banks['retained'][i,:a],banks['shuffled'][i,:a])
            np.testing.assert_array_equal(banks['retained'][i,b:],banks['shuffled'][i,b:])
            self.assertLessEqual(lengths['retained'][i],128)
            self.assertEqual(banks['retained'][i,lengths['retained'][i]-1],1)
        self.assertEqual(donors[7],7)
        self.assertEqual(audit['placebo_changed_items'],6)
        self.assertEqual(set(donors[1:7]),set(range(1,7)))
        self.assertEqual(lengths['retained'][1],128)
        a,b=audit['description_offsets']['1']
        self.assertGreater(b-a,40)
    def test_packing_is_target_free_and_deterministic(self):
        catalog=self.catalog();tok=TinyTokenizer()
        a=pack_catalog(catalog,tok,'Toys')
        catalog['targets']=[99,100]
        b=pack_catalog(catalog,tok,'Toys')
        for name in ('original','retained','shuffled'):np.testing.assert_array_equal(a[0][name],b[0][name])
        np.testing.assert_array_equal(a[2],b[2])


if __name__=='__main__':unittest.main()
